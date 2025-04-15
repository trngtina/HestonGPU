#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define N_TRAJ (1 << 16)
#define N_TRIPLETS 1000

typedef struct {
    float kappa;
    float theta;
    float sigma;
    float rho;
} HestonParam;

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__device__ float rgamma(curandState *state, float alpha) {
    if (alpha < 1.0f) {
        float u = curand_uniform(state);
        return rgamma(state, alpha + 1.0f) * powf(u, 1.0f / alpha);
    }
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    float z, u, x, v;
    while (true) {
        z = curand_normal(state);
        u = curand_uniform(state);
        x = 1.0f + c * z;
        v = x * x * x;
        if (z > -1.0f / c && logf(u) < (0.5f * z * z + d - d * v + d * logf(v)))
            return d * v;
    }
}

__global__ void init_curand_state_k(curandState* state) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(0, idx, 0, &state[idx]);
}

__global__ void MC_Heston_Exact_kernel(float S_0, float v_0, float r, float sqrt_dt, float K, int N, curandState *state, float *d_sum, float *d_sum2, int n_triplets, HestonParam *d_params) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_triplets * N_TRAJ;
    if (gid >= total_threads) return;
    int triplet_idx = gid >> 16;
    if (triplet_idx >= n_triplets) return;
    HestonParam p = d_params[triplet_idx];
    float kappa = p.kappa, theta = p.theta, sigma = p.sigma, rho = p.rho;
    float sigma2 = sigma * sigma;
    float dt = sqrt_dt * sqrt_dt;
    float d_param = 2.0f * kappa * theta / sigma2;
    float v = v_0, vI = 0.0f, S = S_0;
    curandState localState = state[gid];

    extern __shared__ float A[];
    float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

    for (int i = 0; i < N; i++) {
        float lambda = (2.0f * kappa * expf(-kappa * dt) * v) / (sigma2 * (1.0f - expf(-kappa * dt)));
        int N_pois = curand_poisson(&localState, lambda);
        float gamma = rgamma(&localState, d_param + N_pois);
        vI += 0.5f * dt * v;
        v = (sigma2 * (1.0f - expf(-kappa * dt)) / (2.0f * kappa)) * gamma;
        vI += 0.5f * dt * v;
    }
    float m = -0.5f * vI + (rho / sigma) * (v - v_0 - kappa * theta + kappa * vI);
    float Sigma = sqrtf((1.0f - rho * rho) * vI);
    float2 G = curand_normal2(&localState);
    S = S_0 * expf(m + Sigma * G.x);

    R1s[threadIdx.x] = expf(-r * dt * N) * fmaxf(0.0f, S-K)/N_TRAJ ;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * N_TRAJ;

    __syncthreads();
	int i = blockDim.x / 2;
	while (i!=0){
		if (threadIdx.x < i){
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;	
	}
	
	if (threadIdx.x == 0){
		atomicAdd(&d_sum[triplet_idx], R1s[0]);
		atomicAdd(&d_sum2[triplet_idx], R2s[0]);
	}

    state[gid] = localState;
}

__global__ void MC_Heston_Almost_Exact_kernel(float S_0, float v_0, float r, float sqrt_dt, float K, int N, curandState *state, float *d_sum, float *d_sum2, int n_triplets, HestonParam *d_params) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_triplets * N_TRAJ;
    if (gid >= total_threads) return;
    int triplet_idx = gid >> 16;
    HestonParam p = d_params[triplet_idx];
    float kappa = p.kappa, theta = p.theta, sigma = p.sigma, rho = p.rho;
    float dt = sqrt_dt * sqrt_dt;
    float S = S_0;
    float v = v_0, v_next;
    float k0 = (-rho * kappa * theta / sigma) * dt;
    float k1 = (rho * kappa / sigma - 0.5f) * dt - rho / sigma;
    float k2 = rho / sigma;
    curandState localState = state[gid];

    extern __shared__ float A[];
    float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

    for (int i = 0; i < N; i++){
        float2 G = curand_normal2(&localState);
        v_next = fmaxf(v + kappa*(theta - v)*dt + sigma * sqrtf(v) * sqrt_dt * G.x, 0.0f);
        S = S + k0 + k1 * v + k2 * v_next + sqrtf((1.0f - rho*rho) * v) * sqrt_dt * (rho * G.x + sqrtf(1.0f - rho*rho) * G.y);
        v = v_next;
    }
    R1s[threadIdx.x] = expf(-r * dt * N) * fmaxf(0.0f, S-K)/N_TRAJ ;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * N_TRAJ;

    __syncthreads();
	int i = blockDim.x / 2;
	while (i!=0){
		if (threadIdx.x < i){
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;	
	}
	
	if (threadIdx.x == 0){
		atomicAdd(&d_sum[triplet_idx], R1s[0]);
		atomicAdd(&d_sum2[triplet_idx], R2s[0]);
	}

    state[gid] = localState;
}

__global__ void MC_Heston_Euler_kernel(float S_0, float v_0, float r, float sqrt_dt, float K, int N, curandState *state, float *d_sum, float *d_sum2, int n_triplets, HestonParam *d_params) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_triplets * N_TRAJ;
    if (gid >= total_threads) return;
    int triplet_idx = gid >> 16;
    
    HestonParam p = d_params[triplet_idx];
    float dt = sqrt_dt * sqrt_dt;
    float S = S_0;
    float v = v_0, v_next;
    curandState localState = state[gid];

    extern __shared__ float A[];
    float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

    for (int i = 0; i < N; i++){
        float2 G = curand_normal2(&localState);
        v_next = fmaxf(v + p.kappa*(p.theta - v)*dt + p.sigma * sqrtf(v) * sqrt_dt * G.x, 0.0f);
        S = S + r * S * dt + sqrtf(v) * S * sqrt_dt * (p.rho * G.x + sqrtf(1.0f - p.rho*p.rho) * G.y);
        v = v_next;
    }
    R1s[threadIdx.x] = expf(-r * dt * N) * fmaxf(0.0f, S-K)/N_TRAJ ;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * N_TRAJ;

    __syncthreads();
	int i = blockDim.x / 2;
	while (i!=0){
		if (threadIdx.x < i){
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;	
	}
	
	if (threadIdx.x == 0){
		atomicAdd(&d_sum[triplet_idx], R1s[0]);
		atomicAdd(&d_sum2[triplet_idx], R2s[0]);
	}

    state[gid] = localState;
}

void generateTriplets(HestonParam *h_params) {
    float kappa_vals[10] = {0.5f,0.75f,1.0f,1.25f,1.5f,1.75f,2.0f,2.25f,2.5f,2.75f};
    float theta_vals[10] = {0.05f,0.06f,0.07f,0.08f,0.09f,0.10f,0.11f,0.12f,0.13f,0.14f};
    float sigma_vals[10] = {0.1f,0.12f,0.14f,0.16f,0.18f,0.20f,0.22f,0.24f,0.26f,0.28f};
    float rho_vals[10] = {-0.95f,-0.85f,-0.75f,-0.65f,-0.55f,-0.45f,-0.35f,-0.25f,-0.15f,-0.05f};
    int idx = 0;
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            for (int k = 0; k < 10; k++){
                HestonParam p;
                p.kappa = kappa_vals[i];
                p.theta = theta_vals[j];
                p.sigma = sigma_vals[k];
                p.rho = rho_vals[i];
                h_params[idx++] = p;
            }
        }
    }
}

int main(void) {
    int n_triplets = N_TRIPLETS;
    int total_threads = n_triplets * N_TRAJ;
    int threadsPerBlock = 1024;
    int numBlocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
    float T = 1.0f;
    float S_0 = 1.0f;
    float v_0 = 0.1f;
    float r = 0.0f;
    float K = S_0;
    int N = 1000;
    float sqrt_dt = sqrtf(T / (float)N);
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, total_threads * sizeof(curandState)));
    init_curand_state_k<<<numBlocks, threadsPerBlock>>>(d_states);
    HestonParam *h_params = (HestonParam*)malloc(n_triplets * sizeof(HestonParam));
    generateTriplets(h_params);
    HestonParam *d_params;
    testCUDA(cudaMalloc(&d_params, n_triplets * sizeof(HestonParam)));
    testCUDA(cudaMemcpy(d_params, h_params, n_triplets * sizeof(HestonParam), cudaMemcpyHostToDevice));
    free(h_params);

    
    float *d_sum, *d_sum2;
    testCUDA(cudaMallocManaged(&d_sum, n_triplets * sizeof(float)));
    testCUDA(cudaMallocManaged(&d_sum2, n_triplets * sizeof(float)));
    for (int i = 0; i < n_triplets; i++){
        d_sum[i] = 0.0f;
        d_sum2[i] = 0.0f;
    }
    cudaEvent_t start, stop;
    float time_exact, time_almost, time_euler;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MC_Heston_Exact_kernel<<<numBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(S_0, v_0, r, sqrt_dt, K, N, d_states, d_sum, d_sum2, n_triplets, d_params);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_exact, start, stop);
    cudaDeviceSynchronize();
    printf("d_sum[0] = %f\n", d_sum[0]);
    printf("d_sum2[0] = %f\n", d_sum2[0]);

    HestonParam* h_params_out = (HestonParam*)malloc(n_triplets * sizeof(HestonParam));
    cudaMemcpy(h_params_out, d_params, n_triplets * sizeof(HestonParam), cudaMemcpyDeviceToHost);

    FILE *fpt;
    fpt = fopen("MC_Heston_Exact.csv", "w+");
    fprintf(fpt, "triplet,kappa,theta,sigma,rho,price,err,exec_time\n");
    for (int i = 0; i < n_triplets; i++){
        float price = d_sum[i];
        float err = 1.96 * sqrt((double)(1.0f / (N_TRAJ - 1)) * (N_TRAJ*d_sum2[i] - (d_sum[i] * d_sum[i])))/sqrt((double)N_TRAJ);
        fprintf(fpt, "%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.2f\n", i, h_params_out[i].kappa, h_params_out[i].theta, h_params_out[i].sigma, h_params_out[i].rho, price, err, time_exact/N_TRIPLETS);
    }
    fclose(fpt);


    for (int i = 0; i < n_triplets; i++){
        d_sum[i] = 0.0f;
        d_sum2[i] = 0.0f;
    }

    cudaEventRecord(start, 0);

    MC_Heston_Almost_Exact_kernel<<<numBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(S_0, v_0, r, sqrt_dt, K, N, d_states, d_sum, d_sum2, n_triplets, d_params);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_almost, start, stop);
    cudaDeviceSynchronize();

    fpt = fopen("MC_Heston_Almost_Exact.csv", "w+");
    fprintf(fpt, "triplet,kappa,theta,sigma,rho,price,err,exec_time\n");
    for (int i = 0; i < n_triplets; i++){
        float price = d_sum[i];
        float err = 1.96 * sqrt((double)(1.0f / (N_TRAJ - 1)) * (N_TRAJ*d_sum2[i] - (d_sum[i] * d_sum[i])))/sqrt((double)N_TRAJ);
        fprintf(fpt, "%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.2f\n", i, h_params_out[i].kappa, h_params_out[i].theta, h_params_out[i].sigma, h_params_out[i].rho, price, err, time_almost/N_TRIPLETS);
    }



    fclose(fpt);
    for (int i = 0; i < n_triplets; i++){
        d_sum[i] = 0.0f;
        d_sum2[i] = 0.0f;
    }

    cudaEventRecord(start, 0);
    MC_Heston_Euler_kernel<<<numBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(S_0, v_0, r, sqrt_dt, K, N, d_states, d_sum, d_sum2, n_triplets, d_params);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_euler, start, stop);
    cudaDeviceSynchronize();
    fpt = fopen("MC_Heston_Euler.csv", "w+");
    fprintf(fpt, "triplet,kappa,theta,sigma,rho,price,err,exec_time\n");
    for (int i = 0; i < n_triplets; i++){
        float price = d_sum[i];
        float err = 1.96 * sqrt((double)(1.0f / (N_TRAJ - 1)) * (N_TRAJ*d_sum2[i] - (d_sum[i] * d_sum[i])))/sqrt((double)N_TRAJ);
        fprintf(fpt, "%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.2f\n", i, h_params_out[i].kappa, h_params_out[i].theta, h_params_out[i].sigma, h_params_out[i].rho, price, err, time_euler/N_TRIPLETS);
    }
    fclose(fpt);


    cudaFree(d_states);
    cudaFree(d_params);
    cudaFree(d_sum);
    cudaFree(d_sum2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
