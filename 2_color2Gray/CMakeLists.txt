############################################################################

#CMAKE for OpenCV+CUDA presentation
#Dr. N.R Cejas Bolecek ICNPG2017, June 5th 2017
#reneczechdev@gmail.com
#Adapted from udacity CS344 course: Intro to Parallel Programming

###########################################################################

cmake_minimum_required(VERSION 2.8)

project( icnpg17_opencv_2 )

find_package(OpenCV REQUIRED)
FIND_PACKAGE(CUDA)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)

SET (icnpg17_opencv_2_files main.cpp colour2GraySerial.cpp compare.cpp)

CUDA_ADD_EXECUTABLE(icnpg17_opencv_2 ${icnpg17_opencv_2_files} ${hdr} ${cu})
target_link_libraries( icnpg17_opencv_2 ${OpenCV_LIBS} )