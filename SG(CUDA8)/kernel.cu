#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>


__global__ void Evolve(bool* field, float* scores, double b, int size, bool* next_field) {
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Score
	if (col < size && row < size) {
		float score = 0;

		for (int i = -1; i <= 1; i++) //Row
		{
			for (int j = -1; j <= 1; j++) //Col
			{
				int memberIndex = (col + i + size) % size + size * ((row + j + size) % size);

				if (field[memberIndex] == 1)
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
			int memberIndex = (col + i + size) % size + size * ((row + j + size) % size);

			if (scores[bestStrategyIndex] < scores[memberIndex]) 
			{
				bestStrategyIndex = memberIndex;
			}
		}
	}

	next_field[row*size + col] = field[bestStrategyIndex];
}



void InitField(bool* field, size_t size) {
	for (size_t i = 0; i < size*size; i++) {
		field[i] = rand() % 2;
	}
}

void InitScores(float* scores, size_t size) {
	for (size_t i = 0; i < size*size; i++) {
		scores[i] = 0;
	}
}

void PrintField(bool* field, int size) {
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

void PrintScores(float* scores, size_t size) {
	for (size_t i = -1; i < size; i++) {
		for (size_t j = 0; j < size; j++)
		{
			if (i == -1) 
				printf("_");
			else
				printf("%.1f ", scores[i*size + j]);
		}
		printf("\n");
	}
}

int main()
{
	bool* field;
	size_t size = 10;
	double b = 1.81;

	dim3 block(size, size);

	bool *d_field, *d_next_field;
	float *d_scores;

	field = (bool*)malloc(sizeof(bool)*size*size);
	
	// GPU Memory
	cudaMalloc((void**)&d_field, sizeof(bool)*size*size);
	cudaMalloc((void**)&d_scores, sizeof(float)*size*size);
	cudaMalloc((void**)&d_next_field, sizeof(bool)*size*size);

	InitField(field, size);
	PrintField(field, size);

	for (int i = 0; i < 10; i++) {
		// Init scores with zeros in GPU Memory
		cudaMemcpy(d_field, field, size*size, cudaMemcpyKind::cudaMemcpyHostToDevice);

		cudaMemset(d_scores, 0, size*size);		
	
		//Scores<<<1, block>>>(d_field, size, b, d_scores);
		//Strategy<<<1, block>>>(d_field, d_scores, size, d_next_field);		

		Evolve<<<1, block>>>(d_field, d_scores, b, size, d_next_field);

		cudaMemcpy(field, d_next_field, size*size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		PrintField(field, size);
	}

	cudaFree(d_field);
	cudaFree(d_next_field);
	cudaFree(d_scores);

	getchar();

    return 0;
}
