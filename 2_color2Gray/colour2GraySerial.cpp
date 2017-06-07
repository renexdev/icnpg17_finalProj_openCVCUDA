#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>

void colour2GraySerial(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols)
{
   /* tomar el tiempo inicial */
    struct timeval start;
    gettimeofday(&start, NULL);

  for (size_t r = 0; r < numRows; ++r) {
    for (size_t c = 0; c < numCols; ++c) {
      uchar4 rgba = rgbaImage[r * numCols + c];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[r * numCols + c] = channelSum;
    }
  }

    /* tomar el tiempo final */
    struct timeval finish;
    gettimeofday(&finish, NULL);

    /* imprimir el tiempo transcurrido */
    double time = ((finish.tv_sec - start.tv_sec) * 1000.0) + ((finish.tv_usec - start.tv_usec) / 1000.0);
    printf("Tiempo en CPU: %g ms \n", time);  
}

