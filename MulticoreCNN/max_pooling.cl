__kernel void max_pooling (
    __global float* input,
    __global float* output,
    const int inputNBYN,
    const int outputNBYN
) {
    // 2x2������ �����Ͽ� �ִ밪�� �Ǵ��� output�� �ִ´�.
    // 0, 0�� 
    //  (0, 0) (1, 0)
    //  (0, 1) (1, 1)
    // 0, 1��
    //  (2, 0) (3, 0)
    //  (2, 1) (2, 1)
    // (30, 10) ������ max_pooling�� �Ѵٸ�, 
    //  (60, 20) (61, 20)
    //  (60, 21) (61, 21)
    // �ΰ�?
    const int destX = get_global_id(0);
    const int destY = get_global_id(1);
    const int destZ = get_global_id(2);
    const int srcOffset = inputNBYN * inputNBYN * destZ;
    int srcX = destX << 1;
    int srcY = destY << 1;
    float max = 0.0f;

    if (max < input[srcOffset + inputNBYN * srcY    + srcX])                           
        max = input[srcOffset + inputNBYN * srcY    + srcX];
    if (max < input[srcOffset + inputNBYN * srcY    + ++srcX])                        
        max = input[srcOffset + inputNBYN * srcY    + srcX];                       
    if (max < input[srcOffset + inputNBYN * ++srcY  + srcX])
        max = input[srcOffset + inputNBYN * srcY    + srcX];                       
    if (max < input[srcOffset + inputNBYN * srcY    + --srcX])
        max = input[srcOffset + inputNBYN * srcY    + srcX];

    output[outputNBYN * outputNBYN * destZ + outputNBYN * destY + destX] = max;
}