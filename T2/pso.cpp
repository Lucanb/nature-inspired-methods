#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <functional>
#include <ctime>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

const double lower_bound = -600.0;
const double upper_bound = 600.0;
const int num_dimensions = 10;

class benchMarkFunc {
public:
    int dimensions;

    benchMarkFunc(int dimensions) {
        this->dimensions = dimensions;
    }

    double rastrigin(const std::vector<double>& pos) {
        double sum = 0.0;
        for (size_t i = 0; i < pos.size(); ++i) {
            sum += pos[i] * pos[i] - 10.0 * cos(2 * M_PI * pos[i]);
        }
        return 10 * pos.size() + sum;
    }

    double rosenbrock(const std::vector<double>& pos) {
        double sum = 0.0;
        for (size_t i = 0; i < pos.size() - 1; ++i) {
            sum += 100 * std::pow(pos[i + 1] - pos[i] * pos[i], 2) + std::pow(1 - pos[i], 2);
        }
        return sum;
    }

    double michalewicz(const std::vector<double>& pos, double m = 10) {
        double sum = 0.0;
        for (size_t i = 0; i < pos.size(); ++i) {
            sum -= std::sin(pos[i]) * std::pow(std::sin((i + 1) * pos[i] * pos[i] / M_PI), 2 * m);
        }
        return sum;
    }

    double griewangk(const std::vector<double>& pos) {
        double sum = 0.0;
        double prod = 1.0;
        for (size_t i = 0; i < pos.size(); ++i) {
            sum += pos[i] * pos[i] / 4000.0;
            prod *= cos(pos[i] / sqrt(i + 1));
        }
        return sum - prod + 1.0;
    }

    double computeFunction(int index, const std::vector<double>& pos) {
        switch (index) {
        case 1:
            return rastrigin(pos);
        case 2:
            return rosenbrock(pos);
        case 3:
            return michalewicz(pos);
        case 4:
            return griewangk(pos);
        default:
            return rastrigin(pos);
        }
    }
};

class Particle {
public:
    std::vector<double> cromosome;
    std::vector<double> velocity;
    std::vector<double> best_cromosome;
    double best_element;

    Particle(double v_max, int cromosome_size, std::mt19937& gen, std::uniform_real_distribution<>& dist) {
        cromosome.resize(cromosome_size);
        velocity.resize(cromosome_size);
        best_cromosome.resize(cromosome_size);
        for (int i = 0; i < cromosome_size; i++) {
            cromosome[i] = dist(gen) * (upper_bound - lower_bound) + lower_bound;
            velocity[i] = dist(gen) * 2 * v_max - v_max;
        }
        best_cromosome = cromosome;
        best_element = std::numeric_limits<double>::infinity();
    }

    void update(double v_max, double inertia_weight, double personal_c, double social_c, const std::vector<double>& global_best_pos, std::mt19937& gen, std::uniform_real_distribution<>& dist, benchMarkFunc& benchmark) {
        for (size_t i = 0; i < cromosome.size(); ++i) {
            double r1 = dist(gen);
            double r2 = dist(gen);
            double personal_coefficient = personal_c * r1 * (best_cromosome[i] - cromosome[i]);
            double social_coefficient = social_c * r2 * (global_best_pos[i] - cromosome[i]);
            velocity[i] = inertia_weight * velocity[i] + personal_coefficient + social_coefficient;

            if (velocity[i] > v_max) {
                velocity[i] = v_max;
            } else if (velocity[i] < -v_max) {
                velocity[i] = -v_max;
            }

            cromosome[i] += velocity[i];
            if (cromosome[i] > upper_bound) cromosome[i] = upper_bound;
            if (cromosome[i] < lower_bound) cromosome[i] = lower_bound;
        }

        double cost = benchmark.computeFunction(1, cromosome);
        if (cost < best_element) {
            best_element = cost;
            best_cromosome = cromosome;
        }
    }

    static void particle_swarm_optimization(int func_id, int iterations, int pop_size, double v_max, int cromosome_size, int runs = 30) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);

        std::vector<double> results;
        for (int r = 0; r < runs; ++r) {
            benchMarkFunc benchmark(cromosome_size);

            std::vector<Particle> particles;
            particles.reserve(pop_size);
            for (int i = 0; i < pop_size; ++i) {
                particles.emplace_back(v_max, cromosome_size, gen, dist);
            }

            std::vector<double> best_pos(cromosome_size);
            double best_pos_z = std::numeric_limits<double>::infinity();
            double inertia_weight = 0.5 + (dist(gen)) / 2;
            double personal_c = 2.0;
            double social_c = 2.0;

            for (int iter = 0; iter < iterations; ++iter) {
                for (auto& particle : particles) {
                    particle.update(v_max, inertia_weight, personal_c, social_c, best_pos, gen, dist, benchmark);
                }

                for (const auto& particle : particles) {
                    if (particle.best_element < best_pos_z) {
                        best_pos_z = particle.best_element;
                        best_pos = particle.best_cromosome;
                    }
                }
            }

            results.push_back(best_pos_z);
        }

        double sum = std::accumulate(results.begin(), results.end(), 0.0);
        double mean = sum / results.size();
        double sq_sum = std::inner_product(results.begin(), results.end(), results.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / results.size() - mean * mean);
        double min_val = *std::min_element(results.begin(), results.end());

        std::cout << "After " << runs << " runs:" << std::endl;
        std::cout << "Mean: " << mean << std::endl;
        std::cout << "Standard Deviation: " << stdev << std::endl;
        std::cout << "Global Minimum: " << min_val << std::endl;
    }
};

int main() {
    int dimensions = 10;
    int iterations = 1000;
    int pop_size = 30;
    double v_max = 0.1;
    int runs = 30;

    std::cout << "Optimizing Rastrigin function:" << std::endl;
    Particle::particle_swarm_optimization(1, iterations, pop_size, v_max, dimensions, runs);
    std::cout << "\nOptimizing Rosenbrock function:" << std::endl;
    Particle::particle_swarm_optimization(2, iterations, pop_size, v_max, dimensions, runs);
    std::cout << "\nOptimizing Michalewicz function:" << std::endl;
    Particle::particle_swarm_optimization(3, iterations, pop_size, v_max, dimensions, runs);
    std::cout << "\nOptimizing Griewangk function:" << std::endl;
    Particle::particle_swarm_optimization(4, iterations, pop_size, v_max, dimensions, runs);

    return 0;
}
