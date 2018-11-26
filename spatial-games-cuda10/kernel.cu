#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

#include <sys/timeb.h>

#define BLOCK_SIZE 16

__global__ void Evolve(bool* field, float* scores, double b, int size, bool* next_field) 
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int memberIndex;

	// Score
	if (col >= size || row >= size)
		return;
	
	//printf("(%i, %i)\n", col, row);

	float score = 0;

	for (int i = -1; i <= 1; i++) //Row
	{
		for (int j = -1; j <= 1; j++) //Col
		{
			memberIndex = (col + i + size) % size + size * ((row + j + size) % size);

			if (field[memberIndex] == true)
				score++;
		}
	}

	if (!field[row*size + col])
		scores[row*size + col] = score * b;
	else
		scores[row*size + col] = score;
	
	
	__syncthreads();


	// Strategy
	int bestStrategyIndex = row*size + col;

	for (int i = -1; i <= 1; i++) //Row
	{
		for (int j = -1; j <= 1; j++) //Col
		{				
			memberIndex = (col + i + size) % size + size * ((row + j + size) % size);

			if (scores[bestStrategyIndex] < scores[memberIndex]) 
			{
				bestStrategyIndex = memberIndex;
			}
		}
	}

	next_field[row*size + col] = field[bestStrategyIndex];

	__syncthreads();
}



void InitField(bool* field, size_t size, int persentage) 
{
	for (size_t i = 0; i < size*size; i++) {
		field[i] = rand() % 100 > persentage;
	}
}

void PrintField(bool* field, int size) 
{
	printf("\n");
	for (int i = -1; i < size; i++) {
		for (size_t j = 0; j < size; j++)
		{
			if (i == -1) 
				printf("_");
			else
				printf("%s", field[i*size + j]? " " : "#");
		}
		printf("\n");
	}
}

int GetMilliCount() {
	timeb tb;
	ftime(&tb);
	int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
	return nCount;
}

int GetMilliSpan(int nTimeStart) {
	int nSpan = GetMilliCount() - nTimeStart;
	if (nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}

int main(int argc, char * argv[])
{
	int start;

	// Main program

	bool* field;
	unsigned int size = atoi(argv[1]);
	double b = 1.81;

	unsigned int steps = 0;

	bool *d_field, *d_next_field;
	float *d_scores;

	field = (bool*)malloc(sizeof(bool)*size*size);

	// GPU Memory
	cudaMalloc((void**)&d_field, sizeof(bool)*size*size);
	cudaMalloc((void**)&d_scores, sizeof(float)*size*size);
	cudaMalloc((void**)&d_next_field, sizeof(bool)*size*size);

	InitField(field, size, 21);

	unsigned int grid_rows = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 blockSize;
	dim3 gridSize;

	blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	gridSize = dim3(grid_rows, grid_cols, 1);
	
	start = GetMilliCount();

	for (size_t i = 0; i < 1000 && GetMilliSpan(start) < 1000; i++)
	{
		steps++;
		// Init scores with zeros in GPU Memory		
		cudaMemcpy(d_field, field, size * size, cudaMemcpyKind::cudaMemcpyHostToDevice);

		cudaMemset(d_scores, 0, size * size);

		Evolve<<<gridSize, blockSize>>>(d_field, d_scores, b, size, d_next_field);

		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));

		cudaMemcpy(field, d_next_field, size*size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	}

	printf("[%i, %f, %i]\n", size, GetMilliSpan(start) * 0.001 / steps, steps);

	cudaFree(d_field);
	cudaFree(d_next_field);
	cudaFree(d_scores);

    return 0;
}
