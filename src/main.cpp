#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <cstring>

#include <fstream>
#include <iostream>
#include <chrono>

#include "./anglepair.h"
// #include "./filereader.h"
#include "./sim.cuh"

using std::cout;
using std::string;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;

int main(int argc, char const *argv[]) {

    auto start = steady_clock::now();
    auto stop = steady_clock::now();
    milliseconds duration;

    const int ANGLES_COUNT = 100000;
    const int BIN_COUNT = 360;

    string real_data_path;
    string synthetic_data_path;

    if (argc == 3) {
        real_data_path = argv[1];
        synthetic_data_path = argv[2];

    } else if (argc > 1) {
        cout << "Usage: angcorr [REAL_DATA_PATH] [SYNTHETIC_DATA_PATH]"
             << std::endl;
        exit(EXIT_FAILURE);

    } else {
        real_data_path = "/home/lkvikant/gpu/sim/data/data_100k_arcmin.txt";
        synthetic_data_path =
            "/home/lkvikant/gpu/sim/data/flat_100k_arcmin.txt";
    }

    SimResult result = kernelWrapper(real_data_path, synthetic_data_path,
                                      nullptr, ANGLES_COUNT);

    stop = steady_clock::now();
    duration = duration_cast<milliseconds>(stop - start);

    cout << "GPU program finished " << duration.count() << " milliseconds."
         << std::endl;

    vector<float> omega;

    for (size_t i = 0; i < BIN_COUNT; i++) {
        if (result.RR != 0) {
            omega.push_back((result.DD[i] - 2 * result.DR[i] + result.RR[i]) /
                            static_cast<float>(result.RR[i]));
        }
    }

    std::ofstream DR;
    std::ofstream DD;
    std::ofstream RR;
    std::ofstream omega_file;

    DR.open("./result/DR.txt");
    for (size_t i = 0; i < BIN_COUNT; i++) {
        DR << result.DR[i] << "\n";
    }
    DR.close();

    DD.open("./result/DD.txt");
    for (size_t i = 0; i < BIN_COUNT; i++) {
        DD << result.DD[i] << "\n";
    }
    DD.close();

    RR.open("./result/RR.txt");
    for (size_t i = 0; i < BIN_COUNT; i++) {
        RR << result.RR[i] << "\n";
    }
    RR.close();

    omega_file.open("./result/omega.txt");
    for (size_t i = 0; i < omega.size(); i++) {
        omega_file << omega.at(i) << "\n";
    }
    omega_file.close();

    stop = steady_clock::now();
    duration = duration_cast<milliseconds>(stop - start);

    cout << "Main finished in " << duration.count() << " milliseconds."
         << std::endl;

    return 0;
}
