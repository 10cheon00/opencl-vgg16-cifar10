#define IN_BOUNDS(X, Y, BOUND) ((X) >= 0 && (X) < BOUND && (Y) >= 0 && (Y) < BOUND)


__kernel void conv(
    __global float* input,
    __global float* filter,
    __global float* biases,
    __global float* output,
    const int inDim,                  // 입력층 개수
    const int outDim,                 // 출력층 개수
    const int nbyn                    // 층 크기
) {
    // 기존에 생각한 방식은 한 개의 입력층에서 컨볼루션 연산을 한 후 출력층으로 내보내는 것
    // 그러나 출력층으로 내보내지는건 다수의 입력층에서 컨볼루션 연산을 한 결과가 들어가는 것
    // 따라서 입력층을 전부 순회하여야 출력층으로 갈 결과가 나온다.

    // destX, destY는 출력층의 좌표(NBYN * NBYN),
    // destZ는 출력층의 번호(OUTPUT_DIM)
    // 결과적으로 destX*destY*destZ만큼의 workitem이 생긴다. 

    // 입력층과 출력층의 크기는 같다.
    // 따라서 출력층에서 좌표값을 안다면 ,
    //  입력층에서 어느 부분을 쓸어봐야하는지 알 수 있다.

    // 필터의 크기는 3*3 * 입력층개수 * 출력층개수
    // 입력층을 순회하면서 적용한다.
    // 필터가 적용되어 값이 저장될 출력층의 위치에 따라 다른 필터를 사용한다.
    // 그래서 9 * 입력층크기 * 출력층위치를 더해야함

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
