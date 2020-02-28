#if !defined(ANGLE_PAIR_H)
#define ANGLE_PAIR_H

#include <vector>

struct AnglePairs {
    std::vector<float> alphas;
    std::vector<float> cos_deltas;
    std::vector<float> sin_deltas;
};

struct AnglePairArrays {
    float sin_deltas[100000];
    float cos_deltas[100000];
    float alphas[100000];
};

struct Vec3D {
    float x;
    float y;
    float z;
};

#endif // ANGLE_PAIR_H
