/*
=============================================================================================
N.R Cejas Bolecek ICNPG2017, June 5th 2017
#Adapted from udacity CS344 course: Intro to Parallel Programming
=============================================================================================
//Description:
//Adapted from
//http://answers.opencv.org/question/141967/cuda-canny-edge-detector-is-slower-than-cvcanny/
*/
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <string>
#include <iostream>
#include <stdio.h>

#define ITERATENUMBER 1
#define BILFIL false
#define BILFILCANNY false
#define CANNY true

int main(int argc, char **argv)
{   

    printf("Usage ./icnpg17_opencv_1 [input File] [output path]\n");
    printf("Modify flags in the source code to obtain: Bilateral Filter, Canny or Bilateral Filter + Canny & recompile\n");
    printf("Please change the fileName with one of your Img in data Folder\n");


    //IO Files
    std::string filename ("yourImageInDataFolder");
    std::string inPathName ("./data/");
    std::string outPathName ("./outputs/");

    switch (argc)
    {
    case 1:
    printf("Assuming default values ");
    break;

    case 2:
      printf("Assuming:\n \t [input File] = %s \n",argv[1]);

    filename = std::string(argv[1]);
      break;
    case 3:
    filename = std::string(argv[1]);
    outPathName = std::string(argv[2]);
      break;
    default:
      std::cerr << "Usage ./icnpg17_opencv_1 [input File] [output path]"<< std::endl;
      exit(1);
    break;
    }

    cv::Mat ImageHost = cv::imread(inPathName+filename, 0);

    int rows = ImageHost.rows;
    int cols = ImageHost.cols;
    std::cout << "r: "<<rows<< " c: "<< cols<< std::endl;
    std::cout << "img.size(): : "<<rows*cols<< std::endl;

    //Another way
    //cv::Size s = mat.size();
    //rows = s.height;
    //cols = s.width;


    cv::Mat ImageHostArr[ITERATENUMBER];

    cv::cuda::GpuMat ImageDev;
    cv::cuda::GpuMat ImageDevArr[ITERATENUMBER];

    ImageDev.upload(ImageHost);


    for (int n = 0; n < ITERATENUMBER; n++)
        cv::resize(ImageHost, ImageHostArr[n], cv::Size(), 0.5*(n+1), 0.5*(n+1), CV_INTER_LINEAR);


    for (int n = 0; n < ITERATENUMBER; n++)
        cv::cuda::resize(ImageDev, ImageDevArr[n], cv::Size(), 0.5*(n+1), 0.5*(n+1), CV_INTER_LINEAR); 


    cv::Mat BilateralFHost[ITERATENUMBER];
    cv::cuda::GpuMat BilateralFDev[ITERATENUMBER];

    cv::Mat Detected_EdgesHost[ITERATENUMBER];
    cv::cuda::GpuMat Detected_EdgesDev[ITERATENUMBER];

    std::ofstream File1, File2;

    File1.open(outPathName+"1_"+filename+"_canny_cpu.txt");
    File2.open(outPathName+"1_"+filename+"_canny_gpu.txt");


    std::cout << "Process started... /n" << std::endl;
    for (int n = 0; n < ITERATENUMBER; n++) {
        auto start = std::chrono::high_resolution_clock::now();
        if(BILFIL){
            cv::bilateralFilter(ImageHostArr[n], BilateralFHost[n], -1, 50, 7);
        }
        if(BILFILCANNY){
            cv::bilateralFilter(ImageHostArr[n], BilateralFHost[n], -1, 50, 7);
            cv::Canny(BilateralFHost[n], Detected_EdgesHost[n], 2.0, 100.0, 3, false);
        }
        if(CANNY){

            cv::Canny(ImageHostArr[n], Detected_EdgesHost[n], 2.0, 100.0, 3, false);
        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = finish - start;
        File1 << "Image Size: " << ImageHostArr[n].rows* ImageHostArr[n].cols << "  " << "Elapsed Time: " << elapsed_time.count() * 1000 << " msecs" << "/n" << std::endl;

    }


    if(BILFILCANNY || CANNY){
        cv::Mat dstCpu(Detected_EdgesHost[ITERATENUMBER-1]);
       imwrite(outPathName+"1_"+filename+"_CPU.png", dstCpu);
    }
    if(BILFIL){
        cv::Mat dstCpu(BilateralFHost[ITERATENUMBER-1]);
        imwrite(outPathName+"1_"+filename+"_CPU.png", dstCpu);
    }
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(2.0, 100.0, 3, false);   



    for (int n = 0; n < ITERATENUMBER; n++) {
        auto start = std::chrono::high_resolution_clock::now();
        if(BILFIL){
            std::cout << "BILFIL-CUDA /n" << std::endl;

            cv::cuda::bilateralFilter(ImageDevArr[n], BilateralFDev[n], -1, 50, 7);
        }
        if(BILFILCANNY){
            std::cout << "BILFILCANNY-CUDA /n" << std::endl;

            cv::cuda::bilateralFilter(ImageDevArr[n], BilateralFDev[n], -1, 50, 7);
            canny_edg->detect(BilateralFDev[n], Detected_EdgesDev[n]);
        }
      
        if(CANNY){
            std::cout << "CANNY-CUDA /n" << std::endl;

            canny_edg->detect(ImageDevArr[n], Detected_EdgesDev[n]);   
                     }
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = finish - start;
        File2 << "Image Size: " << ImageDevArr[n].rows* ImageDevArr[n].cols << "  " << "Elapsed Time: " << elapsed_time.count() * 1000 << " msecs" << "/n" << std::endl;
    }
    std::cout << "Process ended... /n" << std::endl;

    if(BILFILCANNY || CANNY){
        cv::Mat dstGpu(Detected_EdgesDev[ITERATENUMBER-1]);
        imwrite(outPathName+"1_"+filename+"_GPU.png", dstGpu);
    }
    if(BILFIL){
        cv::Mat dstGpu(BilateralFDev[ITERATENUMBER-1]);
        imwrite(outPathName+"1_"+filename+"_GPU.png", dstGpu);
    }
    return 0;
}