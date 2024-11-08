__kernel void max_pooling(
    __global float* input,
    __global float* output,
    const int inputNBYN,
    const int outputNBYN,
    __local float* cache
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
    float max = 0.0f;

    // copy global memory to local memory
    // max pooling
    //  32x32x64  => 16x16x64
    //  16x16x128 => 8x8x128
    //  8x8x256   => 4x4x256
    //  4x4x512   => 2x2x512
    //  ^^^^^^^
    //         `- �̰͵��� ĳ���ؾ���
    // 
    // local_size�� outputNBYN * outputNBYN
    // local_size�� ť�� ������ �� ������
    // ���� destX,destY,destZ�� �ش��ϴ� ������ ĳ��
    // 
    const int groupX = get_group_id(0); // groupX�� ������� ��ȣ�� �ǹ�
    int localX = get_local_id(0) << 1; // localX,Y�� ������� ��ǥ�� �ǹ�
    int localY = get_local_id(1) << 1;

    // �� �Է������� inputNBYN * inputNBYN��ŭ ������ ĳ��..
    // �� workitem���� 2x2��ŭ ĳ��
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
