#ifndef COLOUR2GRAYSERIAL_H__
#define COLOUR2GRAYSERIAL_H__

void colour2GraySerial(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols);

#endif