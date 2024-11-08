#include "cnn.h"
#include "utils.h"
#include <time.h>
#include <stdio.h>
#include <CL/cl.h>
#include <math.h>

extern const int INPUT_DIM[];
extern const int OUTPUT_DIM[];
extern const int NBYN[];

cl_platform_id platform;
cl_program program;
cl_context context;
cl_device_id device;
cl_command_queue queue;
cl_int error;

const char* source_codes[3];
size_t code_lengths[3];
cl_kernel conv;
cl_kernel max_pooling;
cl_kernel fc_layer;

cl_mem input;
cl_mem layer[21];
cl_mem weight[21];
cl_mem biases[21];
float* result;

void enqueueConvolution(cl_mem input, cl_mem output, int index);

void enqueueMaxPooling(cl_mem input, cl_mem output, int index);

void enqueueFullyConnectedLayer(cl_mem input, cl_mem output, int index);

void softmax(float* input, int N);

int findmax(float* input, int classNum);

void cnn_init(float* images, float* network, int num_of_image) {
    /************************************************************/
    /*                      Initialization                      */
    /************************************************************/
    float* w[21];
    float* b[21];
    int offset = 0;
    source_codes[0] = get_source_code("conv.cl", &code_lengths[0]);
    source_codes[1] = get_source_code("max_pooling.cl", &code_lengths[1]);
    source_codes[2] = get_source_code("fc_layer.cl", &code_lengths[2]);

    error = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(error);
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(error);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    CHECK_ERROR(error);
    queue = clCreateCommandQueue(context, device, 0, &error);
    CHECK_ERROR(error);

    /************************************************************/
    /*                       Create Buffer                      */
    /************************************************************/
    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32 * 32 * 3, NULL, &error);
    CHECK_ERROR(error);

    result = new float[OUTPUT_DIM[0] * NBYN[0] * NBYN[0]];

    // 레이어 버퍼 생성
    for (int i = 0; i < 21; i++) {
        layer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i], NULL, &error);
        CHECK_ERROR(error);
    }

    // 필터, 바이어스 버퍼 생성
    for (int i = 0; i < 17; i++) {
        if (i == 2 || i == 5 || i == 9 || i == 13) {
            ++i;
            // pooling layer has no weights and biases
        }
        w[i] = network + offset;
        offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
        weight[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 9 * INPUT_DIM[i] * OUTPUT_DIM[i], w[i], &error);
        CHECK_ERROR(error);

        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
        biases[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * OUTPUT_DIM[i], b[i], &error);
        CHECK_ERROR(error);
    }

    for (int i = 18; i < 21; ++i) {
        w[i] = network + offset;
        offset += INPUT_DIM[i] * OUTPUT_DIM[i];
        weight[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_DIM[i] * OUTPUT_DIM[i], w[i], &error);
        CHECK_ERROR(error);

        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
        biases[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * OUTPUT_DIM[i], b[i], &error);
        CHECK_ERROR(error);
    }


    /************************************************************/
    /*                       Create Kernel                      */
    /************************************************************/

    program = clCreateProgramWithSource(context, 3, source_codes, code_lengths, &error);
    CHECK_ERROR(error);

    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    char buildResult[4096];
    size_t size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildResult), buildResult, &size);
    printf("%s\n", buildResult);
    CHECK_ERROR(error);

    conv = clCreateKernel(program, "conv", &error);
    CHECK_ERROR(error);

    max_pooling = clCreateKernel(program, "max_pooling", &error);
    CHECK_ERROR(error);

    fc_layer = clCreateKernel(program, "fc_layer", &error);
    CHECK_ERROR(error);

    //softmax = clCreateKernel(program, "softmax", &error);
    //CHECK_ERROR(error);
}

/*
 * @param images        1차원 RGB 이미지
 * @param network       레이어순으로 나열된 가중치와 바이어스
 * @param labels        결과값에서 가장 높은 값의 인덱스
 * @param confidences   결과값에서 가장 높은 값
 * @param num_images    입력될 이미지 수
 *
 * @return void
 */
void cnn(float* images, float* network, int* labels, float* confidences, int num_images) {

    cnn_init(images, network, num_images);

    time_t start, end;
    start = clock();
    //TODO
    /************************************************************/
    /*                      Enqueue Commands                    */
    /************************************************************/
    // 컨볼루션
    // inDim이 3, outDim이 64라면
    // 입력층 3개와 필터 0번과 컨볼루션 연산을 하고 출력층 0번에 저장
    // 입력층 3개와 필터 1번과 컨볼루션 연산을 하고 출력층 1번에 저장
    // ...

    int offset = 0;
    for (int i = 0; i < num_images; i++) {
        error = clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(float) * 32 * 32 * 3, images + offset, 0, NULL, NULL);
        offset += 32 * 32 * 3;

        enqueueConvolution(input, layer[0], 0);
        clFinish(queue);

        enqueueConvolution(layer[0], layer[1], 1);
        clFinish(queue);

        enqueueMaxPooling(layer[1], layer[2], 2);
        clFinish(queue);


        enqueueConvolution(layer[2], layer[3], 3);
        clFinish(queue);

        enqueueConvolution(layer[3], layer[4], 4);
        clFinish(queue);

        // todo: floating point precision문제?
        enqueueMaxPooling(layer[4], layer[5], 5); 
        clFinish(queue);


        enqueueConvolution(layer[5], layer[6], 6);
        clFinish(queue);

        enqueueConvolution(layer[6], layer[7], 7);
        clFinish(queue);

        enqueueConvolution(layer[7], layer[8], 8);
        clFinish(queue);

        enqueueMaxPooling(layer[8], layer[9], 9);
        clFinish(queue);


        enqueueConvolution(layer[9], layer[10], 10);
        clFinish(queue);

        enqueueConvolution(layer[10], layer[11], 11);
        clFinish(queue);

        enqueueConvolution(layer[11], layer[12], 12);
        clFinish(queue);

        enqueueMaxPooling(layer[12], layer[13], 13);
        clFinish(queue);


        enqueueConvolution(layer[13], layer[14], 14);
        clFinish(queue);

        enqueueConvolution(layer[14], layer[15], 15);
        clFinish(queue);

        enqueueConvolution(layer[15], layer[16], 16);
        clFinish(queue);

        enqueueMaxPooling(layer[16], layer[17], 17);
        clFinish(queue);


        enqueueFullyConnectedLayer(layer[17], layer[18], 18);
        clFinish(queue);

        enqueueFullyConnectedLayer(layer[18], layer[19], 19);
        clFinish(queue);

        enqueueFullyConnectedLayer(layer[19], layer[20], 20);
        clFinish(queue);

        error = clEnqueueReadBuffer(queue, layer[20], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20], result, 0, NULL, NULL);

        // todo: softmax를 병렬화 해야할까?
        softmax(result, OUTPUT_DIM[20]);
        labels[i] = findmax(result, 10);
        confidences[i] = result[labels[i]];
    }

    end = clock();
    printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);
}

void enqueueConvolution(cl_mem input, cl_mem output, int index) {
    error = clSetKernelArg(conv, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 1, sizeof(cl_mem), &weight[index]);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 2, sizeof(cl_mem), &biases[index]);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 3, sizeof(cl_mem), &output);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 4, sizeof(cl_int), &INPUT_DIM[index]);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 5, sizeof(cl_int), &OUTPUT_DIM[index]);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 6, sizeof(cl_int), &NBYN[index]);
    CHECK_ERROR(error);

    size_t conv_work_size[3] = {
        static_cast<size_t>(NBYN[index]),
        static_cast<size_t>(NBYN[index]),
        static_cast<size_t>(OUTPUT_DIM[index])
    };
    error = clEnqueueNDRangeKernel(queue, conv, 3, NULL, conv_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(error);
}

void enqueueMaxPooling(cl_mem input, cl_mem output, int index) {
    error = clSetKernelArg(max_pooling, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 1, sizeof(cl_mem), &output);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 2, sizeof(cl_int), &NBYN[index - 1]);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 3, sizeof(cl_int), &NBYN[index]);
    CHECK_ERROR(error);

    size_t max_pooling_work_size[3] = {
        static_cast<size_t>(NBYN[index]),
        static_cast<size_t>(NBYN[index]),
        static_cast<size_t>(OUTPUT_DIM[index])
    };

    error = clEnqueueNDRangeKernel(queue, max_pooling, 3, NULL, max_pooling_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(error);
}

void enqueueFullyConnectedLayer(cl_mem input, cl_mem output, int index) {
    error = clSetKernelArg(fc_layer, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 1, sizeof(cl_mem), &output);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 2, sizeof(cl_mem), &weight[index]);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 3, sizeof(cl_mem), &biases[index]);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 4, sizeof(cl_int), &INPUT_DIM[index]);

    size_t fc_layer_work_size = static_cast<size_t>(OUTPUT_DIM[index]);

    error = clEnqueueNDRangeKernel(queue, fc_layer, 1, NULL, &fc_layer_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(error);
}

void softmax(float* input, int N) {
    int i;
    float max = input[0];
    for (i = 1; i < N; i++) {
        if (max < input[i]) max = input[i];
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(input[i] - max);
    }
    for (i = 0; i < N; i++) {
        input[i] = exp(input[i] - max) / (sum + 1e-7f);
    }
}

int findmax(float* input, int classNum) {
    int i;
    int maxIndex = 0;
    float max = 0;
    for (i = 0; i < classNum; i++) {
        if (max < input[i]) {
            max = input[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}
