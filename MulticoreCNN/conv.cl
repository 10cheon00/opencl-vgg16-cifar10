#define IN_BOUNDS(X, Y, BOUND) ((X) >= 0 && (X) < BOUND && (Y) >= 0 && (Y) < BOUND)


__kernel void conv(
    __global float* input,
    __global float* filter,
    __global float* biases,
    __global float* output,
    const int inDim,                  // �Է��� ����
    const int outDim,                 // ����� ����
    const int nbyn                    // �� ũ��
) {
    // ������ ������ ����� �� ���� �Է������� ������� ������ �� �� ��������� �������� ��
    // �׷��� ��������� ���������°� �ټ��� �Է������� ������� ������ �� ����� ���� ��
    // ���� �Է����� ���� ��ȸ�Ͽ��� ��������� �� ����� ���´�.

    // destX, destY�� ������� ��ǥ(NBYN * NBYN),
    // destZ�� ������� ��ȣ(OUTPUT_DIM)
    // ��������� destX*destY*destZ��ŭ�� workitem�� �����. 

    // �Է����� ������� ũ��� ����.
    // ���� ��������� ��ǥ���� �ȴٸ� ,
    //  �Է������� ��� �κ��� ��������ϴ��� �� �� �ִ�.

    // ������ ũ��� 3*3 * �Է������� * ���������
    // �Է����� ��ȸ�ϸ鼭 �����Ѵ�.
    // ���Ͱ� ����Ǿ� ���� ����� ������� ��ġ�� ���� �ٸ� ���͸� ����Ѵ�.
    // �׷��� 9 * �Է���ũ�� * �������ġ�� ���ؾ���

    const int destX = get_global_id(0);
    const int destY = get_global_id(1);
    const int destZ = get_global_id(2);
    const int layerSize = nbyn * nbyn;
    const int fOffset = 9 * inDim * destZ;
    int x;
    int y;
    float sum = 0.0f;

    for (int i = 0; i < inDim; i++) {
        x = destX - 1; // zero padding
        y = destY - 1;

        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i];
        x++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 1];
        x++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 2];
        x = destX - 1;
        y++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 3];
        x++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 4];
        x++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 5];
        x = destX - 1;
        y++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 6];
        x++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 7];
        x++;
        if (IN_BOUNDS(x, y, nbyn))
            sum += input[layerSize * i + nbyn * y + x] * filter[fOffset + 9 * i + 8];
    }

    sum += biases[destZ];
    if (sum < 0.0f) {
        sum = 0.0f;
    }
    output[layerSize * destZ + nbyn * destY + destX] = sum;
}
