__kernel void fc_layer(
    __global float* input,
    __global float* output,
    __global float* weight,
    __global float* biases,
    const int nbyn
) {
    const int outIndex = get_global_id(0);
    // 각 입력층(사실 층에 데이터가 1개임)에 가중치곱과 바이어스합을 함
    // 그 전체 합을 출력층에 저장
    // 0~511, 가중치0,바이어스 0의 합 => 0
    // 0~511, 가중치1,바이어스 1의 합 => 1
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