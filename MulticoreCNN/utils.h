#pragma once

#ifndef _UTILS_H_
#define _UTILS_H_

#pragma warning(disable:4996)

#include <stdlib.h>
#include <stdio.h>


#define CHECK_ERROR(error) \
    if (error != CL_SUCCESS) {\
        printf("ERROR [%s:%d]: %s\n", __FILE__, __LINE__, getErrorMessage(error)); \
        exit(EXIT_FAILURE); \
    }
const char* getErrorMessage(int error);

char* get_source_code(const char* file_name, size_t* len);

void compareLayerResult(float* layer, const char* path, int len);

const int INPUT_DIM[] = {
    3, 64,
    64,

    64,128,
    128,

    128, 256, 256,
    256,

    256, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    512
};

const int OUTPUT_DIM[] = {
    64, 64,
    64,

    128, 128,
    128,

    256, 256, 256,
    256,

    512, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    10
};

// CIFAR10�� ����ϱ� ������ 
// �̹��� ����� �ǹ��ϴ� ���� 
// 224x224�� �ƴ϶� 32x32�� �Ǿ���.
const int NBYN[] = {
    32, 32,
    16,

    16, 16,
    8,

    8, 8, 8,
    4,

    4, 4, 4,
    2,

    2, 2, 2,
    1,

    1,
    1,
    1
};

#endif