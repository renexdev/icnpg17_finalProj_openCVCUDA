#include "utils.h"

//Kernel
__global__
void color2GrayKernel(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int idx_x = (blockIdx.x*blockDim.x + threadIdx.x);
  int idx_y = (blockIdx.y*blockDim.y + threadIdx.y);

  if( (idx_x < numRows) && (idx_y < numCols )) {
    uchar4 rgba = rgbaImage[idx_x * numCols + idx_y];
    float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
    greyImage[idx_x * numCols + idx_y] = channelSum;
  }

}

void colour2GrayPara(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //gnuplot
  //pri sqrt(512) = 22.6274169979695
  //so 22,22,1
  const dim3 blockSize(22,22,1);  // Make sure blockSize.x*blockSize.y*blockSize.z <=512
  int nblock_row = numRows/blockSize.x + 1;
  int nblock_col = numCols/blockSize.y + 1;
  const dim3 gridSize(nblock_row, nblock_col, 1);  
  color2GrayKernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}