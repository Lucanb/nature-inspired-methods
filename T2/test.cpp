#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <functional>
#include <ctime>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double michalewicz(const std::vector<double>& pos, double m = 10) {
    double sum = 0.0;
    for (size_t i = 0; i < pos.size(); ++i) {
        sum -= std::sin(pos[i]) * std::pow(std::sin((i + 1) * pos[i] * pos[i] / M_PI), 2 * m);
    }
    return sum;
}

void test_michalewicz(int dimensions) {
    std::vector<double> pos(dimensions);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, M_PI);

    // Randomly generate positions in the domain [0, M_PI]
    for (int i = 0; i < dimensions; ++i) {
        pos[i] = dis(gen);
    }
    pos[0] = 2.20;
    pos[1] = 1.57;
    double result = michalewicz(pos);
    std::cout << "Test for dimension " << dimensions << " : " << result << std::endl;
}

int main() {
    // Test with 5 and 10 dimensions
    test_michalewicz(2);
    // test_michalewicz(5);
    // test_michalewicz(10);

    return 0;
}
