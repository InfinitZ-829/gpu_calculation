#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cmath>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>


#define M 8
#define N 32768
#define PI 3.14159265358979323846
#define THREAD_NUM_IN_BLOCK 32
#define SHARED_SIZE_Y 3

typedef cuDoubleComplex Complex;

extern __shared__ Complex sharedMemory[];

static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in  %s at line %d\n",
			cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

void outArrayToFile(const char* filename_r,const char* filename_i, cuDoubleComplex* array, int L, int K){
  std::ofstream outFile_real(filename_r);
  std::ofstream outFile_img(filename_i);
  if (outFile_real.is_open() && outFile_img.is_open())
  {
    for (int i = 0; i < 8; ++i)
    {
      for (int j = 0; j < (2 * L + 1) * (2 * K + 1); j++)
      {
        outFile_real << array[j].x << " ";
        outFile_img << array[j].y << " ";
      }
      outFile_real << std::endl;
      outFile_img << std::endl;
    }
    
    outFile_real.close();
    outFile_img.close();
    std::cout << "output to file finished...\n";
  }
  
}


__device__ void displayMatrix(int tx, int ty, int judgeA, int judgeB, cuDoubleComplex * matrix,int num_in_row, int roll, int col, const char* name){
  if (tx == judgeA && ty == judgeB)
  {
    printf("\n*********************** %s ***********************\n", name);
    for (int i = 0; i < roll; i++)
    {
      for (int j = 0; j < col; j++)
      {
        printf("(%.12f,%.12f) ",matrix[i * num_in_row + j].x, matrix[i * num_in_row + j].y);
      }
      printf("\n----------------------------------------------------------\n");
    }
    printf("\n************************************************************\n");
  }
}

/*
 * @brief  calculate RD array
 *         using shared memory to complete whole process
 * @param dev_sig       input 8 channel signal
 * @param dev_window_h  input window function
 * @param dev_x_ref     input reference signal
 * @param dev_exp_val   input exp param
 * @param L             input r-d param
 * @param K             input r-d param
 * @param dev_rd_array  output 8 channel r-d result
 */
__global__ void calcu2DArray(cuDoubleComplex* dev_sig, cuDoubleComplex* dev_window_h,
	cuDoubleComplex* dev_x_ref, cuDoubleComplex* dev_exp_val, int L, int K, 
	cuDoubleComplex* dev_rd_array) {

    // block index
    int bx = blockIdx.x;  // 0 ~ 1023
    int by = blockIdx.y;  // 0 ~ 10

    // thread index
    int tx = threadIdx.x; // 0 ~ 31
    int ty = threadIdx.y; // 0 ~ 31

    int global_tid_x = bx * blockDim.x + tx; // 0 ~ N - 1
    int global_tid_y = by * blockDim.y + ty; // 0 ~ gridDim.y - 1

    // printf("global_tid_x:%d \n", global_tid_x);

    int major_dim = (2 * L + 1) > (2 * K + 1) ? (2* L + 1): (2 * K + 1);
    int minor_dim = (2 * L + 1) < (2 * K + 1) ? (2* L + 1): (2 * K + 1);

    if (global_tid_x == 0 && global_tid_y == 0)
    {
      printf("major_dim: %d\n", major_dim);
      printf("minor_dim: %d\n", minor_dim);    
    }

    // create shared memory
    Complex * shared_xref = &sharedMemory[0];
    Complex * shared_exp = &sharedMemory[THREAD_NUM_IN_BLOCK * THREAD_NUM_IN_BLOCK];
    Complex * shared_sig = &sharedMemory[THREAD_NUM_IN_BLOCK * (THREAD_NUM_IN_BLOCK + minor_dim)];
    Complex * shared_window = &sharedMemory[(THREAD_NUM_IN_BLOCK + minor_dim + 8)* THREAD_NUM_IN_BLOCK];

    // displayMatrix(global_tid_x, global_tid_y, 0, 0, dev_sig, N, 1, 10, "dev_sig");
    // displayMatrix(global_tid_x, global_tid_y, 0, 0, dev_exp_val, N, 1, 10, "dev_exp_val");
    // displayMatrix(global_tid_x, global_tid_y, 0, 0, dev_exp_val, N, 1, 10, "dev_exp_val");
    // displayMatrix(global_tid_x, global_tid_y, 0, 0, dev_x_ref, N, 320, 2, "dev_x_ref");

    if (global_tid_x == 0 && global_tid_y == 0)
    {
        printf("calculation between signal_array and window function finished...\n");
    }

    if (global_tid_x == 0 && global_tid_y == 0)
    {
        printf("start load matrix into shared memory...\n");
    }

    // each thread load a block of the convolution kernel into shared memory
    for (int i = ty; i < THREAD_NUM_IN_BLOCK; i += blockDim.y)
    {
      shared_xref[i * THREAD_NUM_IN_BLOCK + tx] = (global_tid_y < major_dim) ? dev_x_ref[global_tid_y * THREAD_NUM_IN_BLOCK + global_tid_x] : make_cuDoubleComplex(0.0, 0.0);

      // displayMatrix(global_tid_x, global_tid_y, 0, 0, shared_xref, THREAD_NUM_IN_BLOCK, 1, 32, "shared_xref");

      if (global_tid_y < 3)
      {
        shared_exp[i * THREAD_NUM_IN_BLOCK + tx] = dev_exp_val[global_tid_y * N + global_tid_x];
      }

      // displayMatrix(global_tid_x, global_tid_y, 0, 0, shared_exp, THREAD_NUM_IN_BLOCK, 32, 32, "shared_exp");

      if (global_tid_y < 8)
      {
        shared_sig[i * THREAD_NUM_IN_BLOCK + tx] = dev_sig[global_tid_y * N + global_tid_x];
      }
                  
      if (global_tid_y < 1)
      {
        shared_window[tx] = dev_window_h[global_tid_y * THREAD_NUM_IN_BLOCK + global_tid_x];
      }
      
      // displayMatrix(global_tid_x, global_tid_y, 64, 0, shared_window, THREAD_NUM_IN_BLOCK, 1, 32, "shared_window");
    }
    // displayMatrix(global_tid_x, global_tid_y, 0, 0, shared_sig, THREAD_NUM_IN_BLOCK, 8, 32, "shared_sig");
    __syncthreads();
     

    if (global_tid_x == 0 && global_tid_y == 0)
    {
        printf("load matrix into shared memory finish...\n");
    }

    if (global_tid_x == 0 && global_tid_y == 0)
    {
        printf("ready to start conv calculation in shared memory...\n");
    }

    Complex sum = make_cuDoubleComplex(0.0, 0.0);
    Complex window_elem = make_cuDoubleComplex(0.0, 0.0);
    Complex exp_elem = make_cuDoubleComplex(0.0, 0.0);
    Complex sig_elem = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < 8; i++)
    {
      for (int j = 0; j < minor_dim; j++)
      {
        for (int k = ty; k < THREAD_NUM_IN_BLOCK; k+=blockDim.y)
        {
          // for (int x = tx; x < THREAD_NUM_IN_BLOCK; x+=blockDim.x)
          // {
          exp_elem = shared_exp[j * THREAD_NUM_IN_BLOCK + tx];
          window_elem = shared_window[tx];
          sig_elem = shared_sig[i * THREAD_NUM_IN_BLOCK + tx];
          Complex xref_elem = shared_xref[k * THREAD_NUM_IN_BLOCK + tx];
          Complex sum_elem = cuCmul(xref_elem, cuCmul(exp_elem, cuCmul(sig_elem, window_elem)));
          sum = cuCadd(sum, sig_elem);
          
          if (global_tid_x < N && global_tid_y < major_dim)
          {
            dev_rd_array[i * minor_dim * major_dim + j * major_dim + ty] = cuCadd(sum, dev_rd_array[i * minor_dim * major_dim + j * major_dim + ty]);
          }
  
          // if (blockIdx.x == 0 && threadIdx.y == 0)
          // {
          //   printf("exp_elem: %.8f , %.8f\n", exp_elem.x, exp_elem.y);
          // }

          // if (blockIdx.x == 0 && blockIdx.y == 0)
          // {
          //   printf("threadIdx.y = %d  sig_elem: %.8f , %.8f\n", global_tid_y, sig_elem.x, sig_elem.y);
          // }
  
          // if (blockIdx.x == 0 && threadIdx.y == 0)
          // {
          //   printf("xref_elem: %.8f , %.8f\n", xref_elem.x, xref_elem.y);
          // }
          
          // if (blockIdx.x == 0 && threadIdx.y == 0)
          // {
          //   printf("sum_elem: %.16f , %.16f\n", sum_elem.x, sum_elem.y);
          // }  

          // if (blockIdx.x == 0 && threadIdx.y == 0)
          // {
          //   printf("  __thread_x__thread_y:%d, %d  sum: %.8f , %.8f", global_tid_x, global_tid_y, sum.x, sum.y);
          // }
        // }
      }
      }
    }
    if (global_tid_x == 0 && global_tid_y == 0)
    {
      printf("conv calculation in shared memory finished...\n");
    }
}


cuDoubleComplex* initVec(double* real, double* imag, int m, int n, bool flag) {
	cuDoubleComplex* array = new cuDoubleComplex[m * n];
  if (flag)
  {
    for (int i = 0; i < m * n; i++) {
      if (imag != NULL) {
        array[i] = make_cuDoubleComplex(real[i], imag[i]);
      } else {
        array[i] = make_cuDoubleComplex(real[i], 0);
      }
	  }
  } else {
    for (int i = 0; i < m * n; i++) {
      array[i].x = real[i];
      if (imag != NULL) {
        array[i].y = imag[i];
      } else {
        array[i].y = 0;
      }
	  }
  }
  return array;
}
  
cuDoubleComplex* initVec(double* real, unsigned int m, unsigned int n) {
	cuDoubleComplex* array = new cuDoubleComplex[m * n];
	for (int i = 0; i < m * n; i++) {
		array[i].x = real[i];
		array[i].y = 0;
	}
	return array;
}

void calcuTwoDimRd(double* sig, double* sig_imag,
	double* window_h,
	double* x_ref, double* x_ref_imag,
	unsigned int sig_m, unsigned int sig_n,
	unsigned int window_m, unsigned int window_n,
	unsigned int x_ref_m, unsigned int x_ref_n,
	int L, int K) {
	// make CPU varis
	cuDoubleComplex* sig_vec = initVec(sig, sig_imag, sig_m, sig_n, true);
	cuDoubleComplex* window_vec = initVec(window_h, window_m, window_n);
	cuDoubleComplex* x_ref_vec = initVec(x_ref, x_ref_imag, x_ref_m, x_ref_n, false);
  // for (int i = 0; i < THREAD_NUM_IN_BLOCK; i++)
  // {
  //   printf("x_ref_vec: %.8f, %.8f \n", x_ref_vec[i].x, x_ref_vec[i].y);
  // }
  

	// init exp_val xref_val
	cuDoubleComplex* exp_val = new cuDoubleComplex[(2 * K + 1) * N];
	cuDoubleComplex* xref_val = new cuDoubleComplex[(2 * L + 1) * N];

	for (int i = -K; i <= K; i++) {
		for (int n = 0; n < N; n++) {
      double angle = static_cast<double>(2.0 * PI * i * (n + 1) / N);
			exp_val[n + (i + K) * N] = make_cuDoubleComplex(cos(angle), sin(angle));
		}
	}

	for (int i = -L; i <= L; i++) {
		for (int n = 0; n < N; n++) {
			if (i <= 0) {
				if ((n - i) <= N) {
					xref_val[(i + L) * N + n] = x_ref_vec[n - i];
				} else {
					xref_val[(i + L) * N + n].x = 0;
					xref_val[(i + L) * N + n].y = 0;
				}
			}
			if (i > 0) {
				if ((n - i) >= 0) {
					xref_val[(i + L) * N + n] = x_ref_vec[n - i];
				} else {
					xref_val[(i + L) * N + n].x = 0;
					xref_val[(i + L) * N + n].y = 0;
				}
			}
		}
	}

	// define GPU varis
	cuDoubleComplex* dev_sig, * dev_x_ref, * dev_window_h, * dev_rd_array, * dev_exp_val;

	// malloc mems for GPU varis;
	HANDLE_ERROR(cudaMalloc((void**)&dev_sig, 8 * N * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_x_ref, (2 * L + 1) * N * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_window_h, N * sizeof(cuDoubleComplex)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_exp_val, (2 * K + 1) * N * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_rd_array, 8 * (2 * L + 1) * (2 * K + 1) * sizeof(cuDoubleComplex)));

	// copy data to GPU
	HANDLE_ERROR(cudaMemcpy(dev_sig, sig_vec, 8 * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_x_ref, xref_val, (2 * L + 1) * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_window_h, window_vec, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_exp_val, exp_val, (2 * K + 1) * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  int grid_y = ((2 * L + 1) > (2 * K + 1)) ? (2 * L + 1) : (2 * K + 1);
  
	// set kernel
  dim3 threads(THREAD_NUM_IN_BLOCK, THREAD_NUM_IN_BLOCK);
  dim3 grids((N + threads.x - 1) / threads.x, (grid_y + threads.y - 1) / threads.y);

	printf("threadNumInBlock: %d * %d \n", threads.x, threads.y);
	printf("grid demension %d * %d \n", grids.x, grids.y);
  
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start;
  HANDLE_ERROR(cudaEventCreate(&start));

  cudaEvent_t stop;
  HANDLE_ERROR(cudaEventCreate(&stop));

  // Record the start event
  HANDLE_ERROR(cudaEventRecord(start, NULL));

  std::cout << "ready to enter the kernel" << std::endl;
  // kernel start 
	calcu2DArray <<< grids, threads, 
  THREAD_NUM_IN_BLOCK * (THREAD_NUM_IN_BLOCK * 3)* sizeof(Complex)
	>>> (dev_sig, dev_window_h, dev_x_ref, dev_exp_val, L, K, dev_rd_array);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr == cudaSuccess)
		printf("kernel launch success!\n");
	else {
		printf("kernel launch failed with error \"%s\".\n",
		cudaGetErrorString(cudaerr));
	}

  // Record the stop event
  HANDLE_ERROR(cudaEventRecord(stop, NULL));

  // Wait for the stop event to complete
  HANDLE_ERROR(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal  / 1;
  // double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
  //                            static_cast<double>(dimsA.y) *
  //                            static_cast<double>(dimsB.x);
  // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
  //                    (msecPerMatrixMul / 1000.0f);

  printf(
    "Time= %.3f msec," \
    " WorkgroupSize= %u threads/block\n",
    msecPerMatrixMul,
    threads.x * threads.y);

  cuDoubleComplex* out = new cuDoubleComplex[8 * (2 * L + 1) * (2 * K + 1)];
	HANDLE_ERROR(cudaMemcpy(out, dev_rd_array, 8 * (2 * L + 1) * (2 * K + 1) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));	

  printf("output:\n");
  for (int i = 0; i < 1; ++i)
  {
    for (int j = 0; j < 20; ++j)
    {
      printf("(%.16f,%.16f) \n",out[i * N + j].x, out[i * N + j].y);
    }
    printf("\n");
  }

  outArrayToFile("rd_output_real.txt", "rd_output_img.txt", out, 163, 1);
  

	cudaFree(dev_rd_array);
	cudaFree(dev_sig);
	cudaFree(dev_window_h);
	cudaFree(dev_x_ref);
	cudaFree(out);
	free(window_vec);
	free(x_ref_vec);
	free(sig_vec);
}

cuDoubleComplex* getVectFromFile(char* real_, char* imag_, int m, int n) {
	std::ifstream fileReal(real_);
	std::ifstream fileImag(imag_);
	cuDoubleComplex* out = new cuDoubleComplex[m * n];
	int i = 0;
	double real, imag;
	while (fileReal >> real || fileImag >> imag) {
		if (i >= m * n) {
			break;
		}
		out[i].x = real;
		out[i].y = imag;
		i++;
	}
	return out;
}

cuDoubleComplex* getVectFromFile(char* real_, int m, int n) {
	std::ifstream fileReal(real_);
	cuDoubleComplex* out = new cuDoubleComplex[m * n];
	int i = 0;
	double real;
	while (fileReal >> real) {
		if (i >= m * n) {
			break;
		}
		out[i].x = real;
		out[i].y = 0;
		i++;
	}
	std::cout << i << std::endl;
	return out;
}

double* getDoubleFromFile(const char* filename_, int m, int n) {
	std::ifstream fileIn(filename_);
	double* out = new double[m * n];
	double val;
	int i = 0;
	while (fileIn >> val) {
		if (i >= m * n) {
			break;
		}
		out[i] = val;
		i++;
	}
	printf("%s :", filename_);
	std::cout << i << std::endl;
	return out;
}


int main() {
	double* sig_real = getDoubleFromFile("sig_real.txt", 8, 32768);
	double* sig_imag = getDoubleFromFile("sig_imag.txt", 8, 32768);
	double* x_real = getDoubleFromFile("y_real.txt", 1, 32768);
	double* x_imag = getDoubleFromFile("y_imag.txt", 1, 32768);
	double* window_real = getDoubleFromFile("window_h.txt", 1, 32768);
	calcuTwoDimRd(sig_real, sig_imag, window_real, x_real, x_imag, 8, 32768, 1, 32768, 1, 32768, 163, 1);
	return 0;
}

