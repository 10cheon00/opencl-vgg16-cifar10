__kernel void max_pooling(
    __global float* input,
    __global float* output,
    const int inputNBYN,
    const int outputNBYN,
    __local float* cache
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
    float max = 0.0f;

    // copy global memory to local memory
    // max pooling
    //  32x32x64  => 16x16x64
    //  16x16x128 => 8x8x128
    //  8x8x256   => 4x4x256
    //  4x4x512   => 2x2x512
    //  ^^^^^^^
    //         `- 이것들을 캐싱해야함
    // 
    // local_size는 outputNBYN * outputNBYN
    // local_size는 큐에 삽입할 때 정해짐
    // 현재 destX,destY,destZ에 해당하는 데이터 캐싱
    // 
    const int groupX = get_group_id(0); // groupX는 출력층의 번호를 의미
    int localX = get_local_id(0) << 1; // localX,Y는 출력층의 좌표를 의미
    int localY = get_local_id(1) << 1;

    // 한 입력층에서 inputNBYN * inputNBYN만큼 데이터 캐싱..
    // 각 workitem마다 2x2만큼 캐싱
    cache[inputNBYN * localY + localX] = input[srcOffset + inputNBYN * localY + localX];
    localX++;
    cache[inputNBYN * localY + localX] = input[srcOffset + inputNBYN * localY + localX];
    localY++;
    cache[inputNBYN * localY + localX] = input[srcOffset + inputNBYN * localY + localX];
    localX--;
    cache[inputNBYN * localY + localX] = input[srcOffset + inputNBYN * localY + localX];
    localY--;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (max < cache[inputNBYN * localY + localX])
        max = cache[inputNBYN * localY + localX];
    if (max < cache[inputNBYN * localY + ++localX])
        max = cache[inputNBYN * localY + localX];
    if (max < cache[inputNBYN * ++localY + localX])
        max = cache[inputNBYN * localY + localX];
    if (max < cache[inputNBYN * localY + --localX])
        max = cache[inputNBYN * localY + localX];

    output[outputNBYN * outputNBYN * destZ + outputNBYN * destY + destX] = max;
}
