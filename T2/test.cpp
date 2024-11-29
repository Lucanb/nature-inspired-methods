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

// Funcția Michalewicz
double michalewicz(const std::vector<double>& pos, double m = 10) {
    double sum = 0.0;
    for (size_t i = 0; i < pos.size(); ++i) {
        sum -= std::sin(pos[i]) * std::pow(std::sin((i + 1) * pos[i] * pos[i] / M_PI), 2 * m);
    }
    return sum;
}

// Funcția de test pentru funcția Michalewicz
double test_michalewicz(int dimensions) {
    std::vector<double> pos(dimensions);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, M_PI);  // Corectare: intervalul [0, π]

    // Generarea aleatorie a pozițiilor în intervalul [0, π]
    for (int i = 0; i < dimensions; ++i) {
        pos[i] = dist(gen);  // Folosim dist(gen) pentru a obține valori între 0 și π
    }

    double result = michalewicz(pos);
    std::cout << "Test for dimension " << dimensions << " : " << result << std::endl;
    return result;
}
std::vector<double> vec;
int main() {
    // Testează funcția pentru dimensiunile 2 și 5
    int count = 0;
    for (int i = 0; i < 1000000; ++i) {
        if(test_michalewicz(5)< -4.0){
            std::cout<<"BINGO!";
            count++;
        }  // Poți dezactiva comentariile pentru 10 dacă vrei să testezi și pentru dimensiunea 10
        
        // test_michalewicz(10);
    }
    std::cout<<count;
    return 0;
}
