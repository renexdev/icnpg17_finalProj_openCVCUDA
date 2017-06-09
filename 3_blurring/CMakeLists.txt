############################################################################

#CMAKE for OpenCV+CUDA presentation
#Dr. N.R Cejas Bolecek ICNPG2017, June 5th 2017
#reneczechdev@gmail.com
#Adapted from udacity CS344 course: Intro to Parallel Programming

###########################################################################
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# collect source files
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (icnpg17_opencv_3_files main.cpp blurSerial.cpp compare.cpp)

CUDA_ADD_EXECUTABLE(icnpg17_opencv_3 ${icnpg17_opencv_3_files} ${hdr} ${cu})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(icnpg17_opencv_3 ${OpenCV_LIBS})

