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

    // input ĳ��
    for (int i = localIndex; i < inDim; i += localSize) {
        icache[i] = input[i];
    }

    // weight ĳ��
    // �׷� [0, 32)
    // ���� [0, 16)
    // ������ ������ weight ũ�� = 512 * 16 * 4 = 32768 < 49152 (16���θ� ����)
    // �׷� [0]
    // ���� [0, 10)
    // ������ ������ weight ũ�� = 512 * 10 * 4 = 10240
    // �����δ� 512 * localSize�� �����ϰԵ�
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
