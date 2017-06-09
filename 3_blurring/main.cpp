/*
=============================================================================================
N.R Cejas Bolecek ICNPG2017, June 5th 2017
#Adapted from udacity CS344 course: Intro to Parallel Programming
=============================================================================================
//Description:

// Image Blurring
//
// Imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

*/
#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include "blurSerial.h"
#include "compare.h"
#include "complFn.cpp"


void blurPara(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth);


int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

  float *h_filter;
  int    filterWidth;
  std::string input_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;

  printf("Usage ./icnpg17_opencv_2 [input File] [output path]\n");
  printf("Please change the fileName with one of your Img in data Folder\n");

  //IO Files
  std::string fileName ("yourImageInDataFolder");
  std::string inPathName ("./data/");
  std::string outPathName ("./outputs/");
  switch (argc)
  {
  case 1:
    printf("Assuming default values ");
    input_file = (inPathName + fileName);
    output_file = outPathName + "2_" + fileName+"_gpu.png";
    reference_file = outPathName+  "2_" + fileName+"_cpu.png";
    break;

  case 2:
    printf("Assuming:\n \t [input File] = %s \n",argv[1]);

    input_file = inPathName + std::string(argv[1]);
    output_file = outPathName+ "3_" +  std::string(argv[1])+"_gpu.png";
    reference_file = outPathName+  "3_" + std::string(argv[1])+"_cpu.png";
    break;
  case 3:
    input_file = inPathName + std::string(argv[1]);
    output_file = std::string(argv[2])+  "3_" + std::string(argv[1])+"_gpu.png";
    reference_file = std::string(argv[2])+  "3_" + std::string(argv[1])+"_cpu.png";
    break;
  default:
      std::cerr << "Usage ./icnpg17_opencv_3 [input File] [output path]"<< std::endl;
      exit(1);
    break;
  }

  //load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
             &h_filter, &filterWidth, input_file);

  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
  GpuTimer timer;
  timer.Start();

  blurPara(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                     d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("GPU: %f ms.\n", timer.Elapsed());

  if (err < 0) {
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the blurred image

  size_t numPixels = numRows()*numCols();
  //copy the output back to the host
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

  postProcess(output_file, h_outputImageRGBA);

  blurSerial(h_inputImageRGBA, h_outputImageRGBA,
                       numRows(), numCols(),
                       h_filter, filterWidth);

  postProcess(reference_file, h_outputImageRGBA);

    //  Cheater easy way with OpenCV
    //generateReferenceImage(input_file, reference_file, filterWidth);

  compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError,outPathName);

  checkCudaErrors(cudaFree(d_redBlurred));
  checkCudaErrors(cudaFree(d_greenBlurred));
  checkCudaErrors(cudaFree(d_blueBlurred));

  cleanUp();

  return 0;
}
