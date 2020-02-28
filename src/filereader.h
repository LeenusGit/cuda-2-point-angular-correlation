#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "./anglepair.h"

using std::string;
using std::vector;
using std::cout;

#define ARCMIN_2_RAD 0.000290888f
#define ARCMIN_2_DEGREE 0.0166667f

AnglePairs readAngles(const string &path, const size_t &pairs_count) {

    AnglePairs data;
    string line;
    std::ifstream file(path);

    if (!file.good()) {
        std::cout << "Could not open file " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    data.alphas.reserve(pairs_count);
    data.cos_deltas.reserve(pairs_count);
    data.sin_deltas.reserve(pairs_count);

    float alpha;
    float delta;
    float cos_delta;
    float sin_delta;

    while (std::getline(file, line)) {

        std::istringstream iss(line);

        iss >> alpha >> delta;

        delta *= ARCMIN_2_RAD;

        sin_delta = sinf(delta);
        data.sin_deltas.push_back(sin_delta);

        cos_delta = cosf(delta);
        data.cos_deltas.push_back(cos_delta);

        alpha *= ARCMIN_2_RAD;
        data.alphas.push_back(alpha);
    }

    return data;
}

AnglePairArrays readAngleArrays(const string &path, const size_t &pairs_count) {

    AnglePairArrays data;
    string line;
    std::ifstream file(path);

    if (!file.good()) {
        std::cout << "Could not open file " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    float delta;
    float sin_delta;
    float cos_delta;
    float alpha;

    int idx = 0;

    while (std::getline(file, line)) {

        std::istringstream iss(line);

        iss >> alpha >> delta;
        delta *= ARCMIN_2_RAD;

        sin_delta = sinf(delta);
        data.sin_deltas[idx] = sin_delta;

        cos_delta = cosf(delta);
        data.cos_deltas[idx] = cos_delta;

        alpha *= ARCMIN_2_RAD;
        data.alphas[idx] = alpha;

        ++idx;
    }

    return data;
}

vector<Vec3D> readSphericalCoords(const string &path,
                                  const size_t &pairs_count) {

    vector<Vec3D> points;
    points.reserve(pairs_count);

    Vec3D point;

    string line;
    std::ifstream file(path);

    if (!file.good()) {
        std::cout << "Could not open file " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    float alpha;
    float delta;
    float theta;
    int space_idx;

    while (std::getline(file, line)) {

        space_idx = line.find_first_of(" \t");

        alpha = stof(line.substr(0, space_idx));
        delta = stof(line.substr(space_idx));

        alpha *= ARCMIN_2_RAD;

        theta = 5400 - delta;
        theta *= ARCMIN_2_RAD;

        point.x = sinf(theta) * cosf(alpha);
        point.y = sinf(theta) * sinf(alpha);
        point.y = sinf(theta) * sinf(alpha);
        point.z = cosf(theta);

        points.push_back(point);
    }

    return points;
}

#endif // FILEREADER_H_
