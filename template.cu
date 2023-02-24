#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__global__ void convolution(float* inputImage, const float* /* __restrict__ */ mask, 
			    float* outputImage, int imageChannels, int imageWidth,
			    int imageHeight) {
	// load important variables into registers for quick access
	int threadX = threadIdx.x; int threadY = threadIdx.y;
	int blockX = blockIdx.x; int blockY = blockIdx.y;
	int blockDimX = blockDim.x; int blockDimY = blockDim.y;
	int numRows = imageHeight; int numColumns = imageWidth;
	int numChannels = imageChannels;

	//calculate current row, column, and 1-D index
	int currRow = blockDimY * blockY * threadY;
	int currColumn = blockDimX * blockX * threadX;

	//POINTS TO R VALUE OF CURRENT INDEX!!!
	int index = (currColumn * numColumns + currRow) * numChannels;

	// allocate shared memory for tile, 3 represents number of channels
	// __shared__ float currTile[TILE_WIDTH][TILE_WIDTH][3];
	// check if current element is OOB- do nothing
	if(currRow < numRows && currColumn < numColumns) {
		// load RGB values of element into tile
		// for(int k = 0; k < numChannels; k++) {
		//	currTile[threadX][threadY][k] = inputImage[index + k];
		//}
		//make sure entire tile is loaded before computing convolution.
		//__syncthreads();

		// calculate final value of element
		for(int k = 0; k < numChannels; k++) {
			float finalVal = 0;
			for(int x = -1 * (Mask_radius); x < Mask_radius; x++) {
				for(int y = -1 * (Mask_radius); y < Mask_radius; y++) {
					// current element in imageData we are looking at
					int indexColumn = currColumn + x;
					int indexRow = currRow + y;
					float currVal = 0;
					// check if element is halo element
					if(indexRow < numRows && indexColumn < numColumns && indexRow >= 0 && indexColumn >= 0) {
						/*
						// if indexColumn, indexRow are on current tile, load from shared memory
						if(indexRow / TILE_WIDTH == currRow / TILE_WIDTH &&
						   indexColumn / TILE_WIDTH == currColumn / TILE_WIDTH) {
							currVal = currTile[indexColumn % TILE_WIDTH][indexRow % TILE_WIDTH][k];
						// else load from global memory
						} else {
							currVal = inputImage[(indexColumn * numColumns + indexRow) * numChannels + k];
						}
						*/
						currVal = inputImage[(indexColumn * numColumns + indexRow) * numChannels + k];
					}
					finalVal += currVal * mask[(x + Mask_radius) * Mask_width + y + Mask_radius];
				}
			}
			// output must be between 0 and 1
			outputImage[index + k] = clamp(finalVal);
		}
	}

}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage   = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //cudaMalloc input image, output image, input mask file
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float)); 
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE // cudaMemcpy device input, output SYMBOL COPY MASK
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutputImageData, hostOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpyToSymbol(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float));
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(imageWidth / TILE_WIDTH + 1, imageHeight / TILE_WIDTH + 1);

  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                    deviceOutputImageData, imageChannels,
                                    imageWidth, imageHeight);
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
 
  cudaFree(deviceInputImageData); cudaFree(deviceOutputImageData); cudaFree(deviceMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
