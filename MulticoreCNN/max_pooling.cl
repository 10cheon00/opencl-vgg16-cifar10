__kernel void max_pooling (
    __global float* input,
    __global float* output,
    const int inputNBYN,
    const int outputNBYN
) {
    // 2x2공간을 접근하여 최대값을 판단해 output에 넣는다.
    // 0, 0은 
    //  (0, 0) (1, 0)
    //  (0, 1) (1, 1)
    // 0, 1은
    //  (2, 0) (3, 0)
    //  (2, 1) (2, 1)
    // (30, 10) 공간에 max_pooling을 한다면, 
    //  (60, 20) (61, 20)
    //  (60, 21) (61, 21)
    // 인가?
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