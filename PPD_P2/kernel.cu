#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_runtime.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

int divCeil(int a, int b)
{
	if ((a % b) != 0)
		return a / b + 1;
	return a / b;
}

__global__ void kernel(unsigned char* imgIn, unsigned char* imgOut, int width, int height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int line = blockIdx.y * blockDim.y + threadIdx.y;

	int Index = (line * width) + (col);

	if ((col < width - 2) && (line < height - 2))
	{
		int i = Index;
		int Gx = imgIn[i] * 1 + imgIn[i + 1] * 0 + imgIn[i + 2] * -1;
		i = ((line + 1) * width) + (col);
		Gx += imgIn[i] * 2 + imgIn[i + 1] * 0 + imgIn[i + 2] * -2;
		i = ((line + 2) * width) + (col);
		Gx += imgIn[i] * 1 + imgIn[i + 1] * 0 + imgIn[i + 2] * -1;

		i = Index;
		int Gy = imgIn[i] * 1 + imgIn[i + 1] * 2 + imgIn[i + 2] * 1;
		i = ((line + 1) * width) + (col);
		Gy += imgIn[i] * 0 + imgIn[i + 1] * 0 + imgIn[i + 2] * 0;
		i = ((line + 2) * width) + (col);
		Gy += imgIn[i] * -1 + imgIn[i + 1] * -2 + imgIn[i + 2] * -1;

		imgOut[Index] = sqrtf(Gx * Gx + Gy * Gy) * 0.25;
	}

	return;
}

extern "C" bool apply_Sobel(cv::Mat * inputImage, cv::Mat * outputImage)
{
	cudaError_t cudaStatus;
	unsigned char* deviceIn;
	unsigned char* deviceOut;
	int BLOCK_SIZE = 32;

	unsigned int imageSize = inputImage->rows * inputImage->cols * sizeof(unsigned char);
	unsigned int gradientSize = inputImage->rows * inputImage->cols * sizeof(unsigned char);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(divCeil(inputImage->cols, BLOCK_SIZE), divCeil(inputImage->rows, BLOCK_SIZE));

	cudaStatus = cudaMalloc(&deviceIn, imageSize);
	cudaStatus = cudaMalloc(&deviceOut, gradientSize);

	cudaStatus = cudaMemcpy(deviceIn, inputImage->data, imageSize, cudaMemcpyHostToDevice);

	kernel << <dimGrid, dimBlock >> > (deviceIn, deviceOut, inputImage->step1(), inputImage->rows);

	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(deviceIn);
		cudaFree(deviceOut);

		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		cudaFree(deviceIn);
		cudaFree(deviceOut);

		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(outputImage->data, deviceOut, gradientSize, cudaMemcpyDeviceToHost);

	cudaFree(deviceIn);
	cudaFree(deviceOut);

	return cudaStatus;
}