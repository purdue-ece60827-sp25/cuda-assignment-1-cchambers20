
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

	//For CPU verif?
	float* hy2 = new float[vectorSize];

	float scale = static_cast<float>(rand()) / RAND_MAX;

	for(int i = 0; i < vectorSize; i++) {
		hx[i] = static_cast<float>(rand()) / RAND_MAX;
		hy[i] = static_cast<float>(rand()) / RAND_MAX;
		hy2[i] = hy[i];

	}

	
	float* dx;
	float* dy;

	cudaMalloc((void**)&dx, bytes);
	cudaMalloc((void**)&dy, bytes);

	cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);

	saxpy_gpu<<< (vectorSize + 255) / 256, 256>>>(dx, dy, scale, vectorSize);

	cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);


	//Do CPU Verification
	int out = 1;
	// saxpy_cpu(hx, hy2, scale, vectorSize);
	// for(int i = 0; i < vectorSize; i++) {
	// 	float val = abs(hy[i] - hy2[i]);
	// 	if(val > .01) {
	// 		out = 0;
	// 		std::cout << "Error at idx: " << i << ", Diff val is: " << val << "\n";
	// 		break;
	// 	}

	// }


	cudaFree(dx);
	cudaFree(dy);
	delete[] hx;
	delete[] hy;
	delete[] hy2;

	std::cout << "GPU Saxpy DONE!\n";


	return out;
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
	int idx = (threadIdx.x + blockDim.x * blockIdx.x);
	if(idx >= pSumSize) return;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), idx, 0, &rng);

	uint64_t hitCount = 0;

	for(int i = 0; i < sampleSize; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		if((x*x + y*y) <= 1.0) hitCount++;

	}
	pSums[idx] = hitCount;




}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {

	int idx = (threadIdx.x + blockDim.x * blockIdx.x);
	if(idx >= reduceSize) return;

	uint64_t sum = 0;
	uint64_t i = idx * (pSumSize/reduceSize);
	uint64_t max_val = (idx + 1) * (pSumSize/reduceSize);
	for(; i < max_val; i++) {
		sum += pSums[i];
	}
	totals[idx] = sum;



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

	//dev variables
	uint64_t* dSum;
	uint64_t* dTotal;

	//host variables
	uint64_t* hTotal = new uint64_t[reduceThreadCount];



	cudaMalloc((void**)&dSum, generateThreadCount * sizeof(uint64_t));
	cudaMalloc((void**)&dTotal, generateThreadCount * sizeof(uint64_t));

	generatePoints<<< (generateThreadCount+255 / 256), 256>>>(dSum, generateThreadCount, sampleSize);

	//kind of my pass through function
	reduceCounts<<< (generateThreadCount+255 / 256), 256>>>(dSum, dTotal, generateThreadCount, reduceThreadCount);

	cudaMemcpy(hTotal, dTotal, reduceThreadCount*sizeof(uint64_t), cudaMemcpyDeviceToHost);


	uint64_t allHits = 0;
	for(uint64_t i = 0; i < reduceThreadCount; i++) allHits += hTotal[i];

	approxPi = (4.0 * allHits) / (generateThreadCount * sampleSize);

	cudaFree(dSum);
	cudaFree(dTotal);

	delete[] hTotal;


	// //      Insert code here
	// std::cout << "Sneaky, you are ...\n";
	// std::cout << "Compute pi, you must!\n";

	return approxPi;
}
