
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = (threadIdx.x + blockDim.x * blockIdx.x);

	if(idx < size) y[idx] = scale * x[idx] + y[idx];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	// std::cout << "Lazy, you are!\n";
	// std::cout << "Write code, you must\n";

	srand(time(NULL));

	size_t bytes = vectorSize * sizeof(float);

	float* hx = new float[vectorSize];
	float* hy = new float[vectorSize];

	float scale = static_cast<float>(rand()) / RAND_MAX;

	for(int i = 0; i < vectorSize; i++) {
		hx[i] = static_cast<float>(rand()) / RAND_MAX;
		hy[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	
	float* dx;
	float* dy;

	cudaMalloc((void**)&dx, bytes);
	cudaMalloc((void**)&dy, bytes);

	cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);

	//saxpy_gpu<<<(vectorSize + 255) / 256, 256>>>(dx, dy, scale, vectorSize);
	saxpy_gpu<<< ceil(vectorSize / 256), 256>>>(dx, dy, scale, vectorSize);

	cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);

	cudaFree(dx);
	cudaFree(dy);
	delete[] hx;
	delete[] hy;



	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
