__kernel void fc_layer(
    __global float* input,
    __global float* output,
    __global float* weight,
    __global float* biases,
    const int nbyn
) {
    const int outIndex = get_global_id(0);
    // �� �Է���(��� ���� �����Ͱ� 1����)�� ����ġ���� ���̾���� ��
    // �� ��ü ���� ������� ����
    // 0~511, ����ġ0,���̾ 0�� �� => 0
    // 0~511, ����ġ1,���̾ 1�� �� => 1
    // ...

    float sum = 0.0f;
    for (int i=0; i < nbyn; i++) {
        sum += input[i] * weight[nbyn * outIndex + i];
    }
    sum += biases[outIndex];
    if (sum < 0.0f) {
        sum = 0.0f;
    }
    output[outIndex] = sum;
}