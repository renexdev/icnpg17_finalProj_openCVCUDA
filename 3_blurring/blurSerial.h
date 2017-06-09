#ifndef BLURSERIAL_H__
#define BLURSERIAL_H__

void blurSerial(const uchar4* const rgbaImage, uchar4 *const outputImage,
                          size_t numRows, size_t numCols,
                          const float* const filter, const int filterWidth);

#endif