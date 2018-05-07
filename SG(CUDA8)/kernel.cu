#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

#include <time.h>

#define BLOCK_SIZE 16

__global__ void Evolve(bool* field, float* scores, double b, int size, bool* next_field) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int memberIndex;

	// Score
	if (col < size && row < size) {
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
	}
	
	
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
}



void InitField(bool* field, size_t size, int persentage) {
	for (size_t i = 0; i < size*size; i++) {
		field[i] = rand() % 100 < persentage;
	}
}

void PrintField(bool* field, int size) {
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

int main()
{
	FILE *file;
	clock_t start, finish;
	char file_name[64];

//// Main program

	bool* field;
	unsigned int size = 32;
	double b = 1.81;

	unsigned int steps = 1;

	bool *d_field, *d_next_field;
	float *d_scores;


	printf("Input field size: ");
	scanf("%i", &size);
	printf("Input b: ");
	scanf("%lf", &b);
	printf("Input number of steps: ");
	scanf("%i", &steps);
	printf("Size = %i\nb = %f\nSteps = %i\n\n", size, b, steps);

	sprintf(file_name, "%i_%0.2f.txt", size, b);

	file = fopen(file_name, "w");

	field = (bool*)malloc(sizeof(bool)*size*size);
	
// GPU Memory
	cudaMalloc((void**)&d_field, sizeof(bool)*size*size);
	cudaMalloc((void**)&d_scores, sizeof(float)*size*size);
	cudaMalloc((void**)&d_next_field, sizeof(bool)*size*size);


	InitField(field, size, 90);

	unsigned int grid_rows = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


	for (int i = 0; i < steps; i++) {
		// Init scores with zeros in GPU Memory		
		start = clock();


		cudaMemcpy(d_field, field, size*size, cudaMemcpyKind::cudaMemcpyHostToDevice);

		cudaMemset(d_scores, 0, size*size);	

		Evolve<<<dimGrid, dimBlock>>>(d_field, d_scores, b, size, d_next_field);

		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));

		cudaMemcpy(field, d_next_field, size*size, cudaMemcpyKind::cudaMemcpyDeviceToHost);


		finish = clock();
		printf("Time for step %i: %f\n", i, ((double)(finish - start)) / CLOCKS_PER_SEC);
		fprintf(file, "%f,", ((double)(finish - start)) / CLOCKS_PER_SEC);
	}

	cudaFree(d_field);
	cudaFree(d_next_field);
	cudaFree(d_scores);

	getchar();

    return 0;
}
