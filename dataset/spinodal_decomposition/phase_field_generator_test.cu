#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define m 128
#define n 128
#define dx 1.0
#define dy 1.0
#define dt 1.0e-2
#define R 8.314
#define M 1.0
#define k 3.57e-1
#define c0 0.60
#define vm 9.8e21

#define BLOCK_SIZE 16
#define THREADS_PER_BLOCK 256
#define NUM_SIMULATIONS 20

FILE *fp1;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// CUDA核函数：生成噪声
__global__ void noise_kernel(double *con, curandState *state, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int i = idx / (n + 2);
    int j = idx % (n + 2);
    
    if (i >= 1 && i <= m && j >= 1 && j <= n) {
        double noise = 0.05;
        double a = curand_uniform_double(&state[idx]);
        con[j + i * (n + 2)] = c0 + noise * (0.5 - a);
    }
}

// CUDA核函数：初始化随机数生成器
__global__ void setup_kernel(curandState *state, unsigned long seed, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// CUDA核函数：周期性边界条件
__global__ void boundary_kernel(double *con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // y方向边界
    if (idx < n + 1) {
        int j = idx;
        if (j >= 1 && j <= n) {
            con[j + 0 * (n + 2)] = con[j + m * (n + 2)];
            con[j + (m + 1) * (n + 2)] = con[j + 1 * (n + 2)];
        }
    }
    
    // x方向边界
    if (idx < m + 1) {
        int i = idx;
        if (i >= 1 && i <= m) {
            con[0 + i * (n + 2)] = con[n + i * (n + 2)];
            con[n + 1 + i * (n + 2)] = con[1 + i * (n + 2)];
        }
    }
    
    // 四个角点
    if (idx == 0) {
        con[0 + 0 * (n + 2)] = con[n + m * (n + 2)];
        con[n + 1 + 0 * (n + 2)] = con[1 + m * (n + 2)];
        con[0 + (m + 1) * (n + 2)] = con[n + 1 * (n + 2)];
        con[n + 1 + (m + 1) * (n + 2)] = con[1 + 1 * (n + 2)];
    }
}

// CUDA核函数：计算化学自由能密度的一阶导
__global__ void calculate_dF_kernel(double *con, double *dcon, double *lap_con, double *dF)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= m && j <= n) {
        int idx = j + i * (n + 2);
        
        double T = 973.15;
        double A0 = 15000.0 + 6.1 * T;
        double A1 = -7600 + 3.55 * T;
        
        double c_val = con[idx];
        if (c_val <= 0.0) c_val = 1.0e-10;
        if (c_val >= 1.0) c_val = 1.0 - 1.0e-10;
        
        dcon[idx] = (R * T * log(c_val / (1.0 - c_val)) + 
                    (1.0 - 2.0 * c_val) * A0 + 
                    (-6.0 * c_val + 6.0 * c_val * c_val + 1.0) * A1) / (R * T);
        
        double c1 = con[j + (i + 1) * (n + 2)];
        double c2 = con[j + (i - 1) * (n + 2)];
        double c3 = con[j + 1 + i * (n + 2)];
        double c4 = con[j - 1 + i * (n + 2)];
        double c5 = con[idx];
        
        lap_con[idx] = (c1 + c2 + c3 + c4 - 4.0 * c5) / (dx * dy);
        
        dF[idx] = dcon[idx] - 2 * k * lap_con[idx];
    }
}

// CUDA核函数：设置dF边界条件
__global__ void boundary_dF_kernel(double *dF)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n + 1) {
        int j = idx;
        if (j >= 1 && j <= n) {
            dF[j + 0 * (n + 2)] = dF[j + m * (n + 2)];
            dF[j + (m + 1) * (n + 2)] = dF[j + 1 * (n + 2)];
        }
    }
    
    if (idx < m + 1) {
        int i = idx;
        if (i >= 1 && i <= m) {
            dF[0 + i * (n + 2)] = dF[n + i * (n + 2)];
            dF[n + 1 + i * (n + 2)] = dF[1 + i * (n + 2)];
        }
    }
    
    if (idx == 0) {
        dF[0 + 0 * (n + 2)] = dF[n + m * (n + 2)];
        dF[n + 1 + 0 * (n + 2)] = dF[1 + m * (n + 2)];
        dF[0 + (m + 1) * (n + 2)] = dF[n + 1 * (n + 2)];
        dF[n + 1 + (m + 1) * (n + 2)] = dF[1 + 1 * (n + 2)];
    }
}

// CUDA核函数：更新浓度场
__global__ void update_concentration_kernel(double *con, double *dF, double *lap_dF)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= m && j <= n) {
        int idx = j + i * (n + 2);
        
        double F1 = dF[j + (i + 1) * (n + 2)];
        double F2 = dF[j + (i - 1) * (n + 2)];
        double F3 = dF[j + 1 + i * (n + 2)];
        double F4 = dF[j - 1 + i * (n + 2)];
        double F5 = dF[idx];
        
        lap_dF[idx] = (F1 + F2 + F3 + F4 - 4.0 * F5) / (dx * dy);
        
        con[idx] = con[idx] + dt * M * lap_dF[idx];
        
        if (con[idx] < 0.0) con[idx] = 0.0;
        if (con[idx] > 1.0) con[idx] = 1.0;
    }
}

// 主机函数：噪声初始化
void noise_cuda(double *d_con, curandState *d_state, unsigned long seed)
{
    int total_size = (m + 2) * (n + 2);
    int blocks = (total_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    setup_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_state, seed, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    noise_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_con, d_state, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 主机函数：计算浓度场
void calculate_Pt_concentration_cuda(double *d_con, double *d_dcon, double *d_lap_con, 
                                   double *d_lap_dF, double *d_dF)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    calculate_dF_kernel<<<gridSize, blockSize>>>(d_con, d_dcon, d_lap_con, d_dF);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int boundary_blocks = ((m > n ? m : n) + 2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    boundary_dF_kernel<<<boundary_blocks, THREADS_PER_BLOCK>>>(d_dF);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    update_concentration_kernel<<<gridSize, blockSize>>>(d_con, d_dF, d_lap_dF);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 输出数据到文件
void output_data(double *h_con, const char* filename)
{
    fp1 = fopen(filename, "w");
    if (fp1 == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }
    
    fprintf(fp1, "VARIABLE= \"x\",\"y\",\"FUNCTION\"\n");
    fprintf(fp1, "ZONE t=\"BIG ZONE\",I= %d,J= %d,F=POINT\n", n, m);
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            double concentration = h_con[j + i * (n + 2)];
            fprintf(fp1, "%f\n", concentration);
        }
    }
    fclose(fp1);
}

// 运行单次模拟
void run_simulation(double *d_con, double *d_dcon, double *d_lap_con, 
                    double *d_lap_dF, double *d_dF, curandState *d_state,
                    double *h_con, size_t size, unsigned long seed, 
                    const char* output_dir, int sim_num)
{
    int step;
    int nstep = 102001;
    int nprint = 10;
    char filename[256];
    
    // 重置数组
    CUDA_CHECK(cudaMemset(d_con, 0, size));
    CUDA_CHECK(cudaMemset(d_dcon, 0, size));
    CUDA_CHECK(cudaMemset(d_lap_con, 0, size));
    CUDA_CHECK(cudaMemset(d_lap_dF, 0, size));
    CUDA_CHECK(cudaMemset(d_dF, 0, size));
    
    printf("\n=== Simulation %d/20: %s (seed=%lu) ===\n", sim_num, output_dir, seed);
    noise_cuda(d_con, d_state, seed);
    
    // 保存初始状态
    sprintf(filename, "%s/con_00000.dat", output_dir);
    CUDA_CHECK(cudaMemcpy(h_con, d_con, size, cudaMemcpyDeviceToHost));
    output_data(h_con, filename);
    
    // 主循环
    for (step = 1; step < nstep; step++) {
        if (step % 10000 == 0) {
            printf("  Step: %d / %d (%.1f%%)\n", step, nstep-1, 100.0*step/(nstep-1));
        }
        
        int boundary_blocks = ((m > n ? m : n) + 2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        boundary_kernel<<<boundary_blocks, THREADS_PER_BLOCK>>>(d_con);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        calculate_Pt_concentration_cuda(d_con, d_dcon, d_lap_con, d_lap_dF, d_dF);
        
        if (step % nprint == 0) {
            sprintf(filename, "%s/con_%05d.dat", output_dir, step/nprint);
            CUDA_CHECK(cudaMemcpy(h_con, d_con, size, cudaMemcpyDeviceToHost));
            output_data(h_con, filename);
        }
    }
    
    printf("  Completed: %d files saved\n", nstep/nprint + 1);
}

int main()
{
    double *h_con, *d_con, *d_dcon, *d_lap_con, *d_lap_dF, *d_dF;
    curandState *d_state;
    char dir_name[64];
    
    clock_t start_time, end_time;
    double cpu_time_used;
    
    printf("=== Phase Field Simulation - Dataset Generator ===\n");
    printf("Grid size: %d x %d\n", m, n);
    printf("Number of simulations: %d\n", NUM_SIMULATIONS);
    printf("Total time steps per simulation: 52000\n");
    printf("Output frequency: every 10 steps\n");
    printf("Files per simulation: 5201\n\n");
    
    // 创建20个输出目录
    printf("Creating directories...\n");
    for (int i = 1; i <= NUM_SIMULATIONS; i++) {
        sprintf(dir_name, "test%d", i);
        mkdir(dir_name, 0755);
        printf("  Created: %s/\n", dir_name);
    }
    
    size_t size = (m + 2) * (n + 2) * sizeof(double);
    
    // 分配主机内存
    h_con = (double *)malloc(size);
    if (h_con == NULL) {
        printf("Failed to allocate host memory\n");
        return -1;
    }
    
    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void **)&d_con, size));
    CUDA_CHECK(cudaMalloc((void **)&d_dcon, size));
    CUDA_CHECK(cudaMalloc((void **)&d_lap_con, size));
    CUDA_CHECK(cudaMalloc((void **)&d_lap_dF, size));
    CUDA_CHECK(cudaMalloc((void **)&d_dF, size));
    CUDA_CHECK(cudaMalloc((void **)&d_state, (m + 2) * (n + 2) * sizeof(curandState)));
    
    printf("\nMemory allocated successfully.\n");
    
    start_time = clock();
    
    // 运行20次模拟，每次使用不同的随机种子
    unsigned long base_seed = 10000;
    for (int i = 1; i <= NUM_SIMULATIONS; i++) {
        sprintf(dir_name, "test%d", i);
        unsigned long seed = base_seed + i * 12345;
        
        run_simulation(d_con, d_dcon, d_lap_con, d_lap_dF, d_dF, d_state,
                       h_con, size, seed, dir_name, i);
    }
    
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("\n========================================\n");
    printf("=== All Simulations Completed ===\n");
    printf("========================================\n");
    printf("Total time: %.2f seconds (%.2f minutes)\n", cpu_time_used, cpu_time_used/60.0);
    printf("Average time per simulation: %.2f seconds\n", cpu_time_used/NUM_SIMULATIONS);
    printf("Total files generated: %d\n", NUM_SIMULATIONS * 5201);
    printf("Directories: test1/ to test%d/\n", NUM_SIMULATIONS);
    
    // 释放内存
    free(h_con);
    CUDA_CHECK(cudaFree(d_con));
    CUDA_CHECK(cudaFree(d_dcon));
    CUDA_CHECK(cudaFree(d_lap_con));
    CUDA_CHECK(cudaFree(d_lap_dF));
    CUDA_CHECK(cudaFree(d_dF));
    CUDA_CHECK(cudaFree(d_state));
    
    printf("Memory released successfully.\n");
    return 0;
}
