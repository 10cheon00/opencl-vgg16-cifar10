__kernel void fc_layer(
    __global float* input,
    __global float* output,
    __global float* weight,
    __global float* biases,
    __local float* icache,
    __local float* wcache,
    int inDim
) {
    const int outputIndex = get_global_id(0);
    const int localIndex = get_local_id(0);
    const int localSize = get_local_size(0);

    // input 캐싱
    for (int i = localIndex; i < inDim; i += localSize) {
        icache[i] = input[i];
    }

    // weight 캐싱
    // 그룹 [0, 32)
    // 로컬 [0, 16)
    // 로컬이 참조할 weight 크기 = 512 * 16 * 4 = 32768 < 49152 (16으로만 가능)
    // 그룹 [0]
    // 로컬 [0, 10)
    // 로컬이 참조할 weight 크기 = 512 * 10 * 4 = 10240
    // 실제로는 512 * localSize만 참조하게됨
    for (int i = 0; i < inDim; i++) {
        wcache[inDim * localIndex + i] = weight[inDim * outputIndex + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int i = 0; i < inDim; i++) {
        sum += icache[i] * wcache[inDim * localIndex + i];
    }
    sum += biases[outputIndex];
    if (sum < 0.0f) {
        sum = 0.0f;
    }
    output[outputIndex] = sum;
}
