/*
=============================================================================================
N.R Cejas Bolecek ICNPG2017, June 5th 2017
#Adapted from udacity CS344 course: Intro to Parallel Programming
=============================================================================================
//Description:

// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this implementation.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.
*/
#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include "colour2GraySerial.h"
#include "compare.h"

void colour2GrayPara(const uchar4 * const h_rgbaImage, 
                            uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, 
                            size_t numRows, size_t numCols);

//include the definitions of the above functions
#include "complFn.cpp"

int main(int argc, char **argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

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
    output_file = outPathName+ "2_" +  std::string(argv[1])+"_gpu.png";
    reference_file = outPathName+  "2_" + std::string(argv[1])+"_cpu.png";
	  break;
	case 3:
    input_file = inPathName + std::string(argv[1]);
    output_file = std::string(argv[2])+  "2_" + std::string(argv[1])+"_gpu.png";
    reference_file = std::string(argv[2])+  "2_" + std::string(argv[1])+"_cpu.png";
	  break;
  default:
      std::cerr << "Usage ./icnpg17_opencv_2 [input File] [output path]"<< std::endl;
      exit(1);
    break;
  }

  //load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  //Parallel Implementation

  GpuTimer timer;
  timer.Start();
  //call parallel fn
  colour2GrayPara(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  int err = printf("Tiempo en GPU: %f ms\n", timer.Elapsed());

  if (err < 0) {
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  size_t numPixels = numRows()*numCols();
  checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

  //check results and output the grey image
  postProcess(output_file, h_greyImage);


  //Serial Implementation
  colour2GraySerial(h_rgbaImage, h_greyImage, numRows(), numCols());

  postProcess(reference_file, h_greyImage);

  //compareImages
  compareImages(reference_file, output_file, useEpsCheck, perPixelError, 
                globalError,outPathName);

  cleanup();

  return 0;
}
