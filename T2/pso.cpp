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

class benchMarkFunc {
public:
    int dimensions;

    benchMarkFunc(int dimensions) {
        this->dimensions = dimensions;
    }

    // double rastrigin(const std::vector<double>& pos) {
    //     double sum = 0.0;
    //     for (size_t i = 0; i < pos.size(); ++i) {
    //         sum += pos[i] * pos[i] - 10.0 * cos(2 * M_PI * pos[i]);
    //     }
    //     return 10 * pos.size() + sum;
    // }
    double rastrigin(const std::vector<double>& pos) {
        double sum = 0.0;
        for (size_t i = 0; i < pos.size(); ++i) {
            if (std::isnan(pos[i]) || std::isinf(pos[i])) {
                std::cerr << "Warning: Invalid value in position " << i << std::endl;
                return std::numeric_limits<double>::infinity();
            }
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
    double lower_bound;
    double upper_bound;

    Particle(double v_max, int cromosome_size, std::mt19937& gen, std::uniform_real_distribution<>& dist,double lower_bound,double upper_bound) {
        cromosome.resize(cromosome_size);
        velocity.resize(cromosome_size);
        this->lower_bound = lower_bound;
        this->upper_bound = upper_bound;
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

 static void print_cromosome(int dimension,std::vector<double> cromosome){
    for(int i =0;i<dimension;i++){
        std::cout<<cromosome[i]<<' ';
    }
    std::cout<<'\n';
 }
//aici scrie rezultatele intr-un fisier separat duma dimensions 5.out ; 10.out ; 30.out

 static void particle_swarm_optimization(int func_id,double personal_c,double social_c, int iterations, int pop_size, double v_max, int cromosome_size, int runs = 30) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);

        std::vector<double> results;
        
        double lower_bound, upper_bound;
        
            switch (func_id) {
        case 1: // Rastrigin
                lower_bound = -5.12;
                upper_bound = 5.12;
            break;
        
        case 2: // Rosenbrock
                lower_bound = -5.0;
                upper_bound = 10.0;
            break;
        
        case 3: // Michalewicz
                lower_bound = 0.0;
                upper_bound = 3.14159265;
            break;

        case 4: // Griewangk
                lower_bound = -600.0;
                upper_bound = 600.0;
            break;

        default:
            std::cerr << "Funcție necunoscută!" << std::endl;
            break;
    }
        std::vector<double> Best;
        for (int r = 0; r < runs; ++r) {
            benchMarkFunc benchmark(cromosome_size);

            std::vector<Particle> particles;
            particles.reserve(pop_size);
            for (int i = 0; i < pop_size; ++i) {
                particles.emplace_back(v_max, cromosome_size, gen, dist, lower_bound, upper_bound);
            }
                
            std::vector<double> best_pos(cromosome_size);
            double best_pos_z = std::numeric_limits<double>::infinity();
            double inertia_weight = 0.5 + (dist(gen)) / 2; //ciudat


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
            Best = best_pos;
            results.push_back(best_pos_z);
        }


        double sum = std::accumulate(results.begin(), results.end(), 0.0);
        double mean = sum / results.size();
        double sq_sum = std::inner_product(results.begin(), results.end(), results.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / results.size() - mean * mean);
        double min_val = *std::min_element(results.begin(), results.end());

        // Alege numele fișierului pe baza dimensiunii cromozomului
        std::string file_name;
        switch (cromosome_size) {
            case 2:
                std::cout<<"----------------------------------------------"<<'\n';
                std::cout<<"RESULTS : "<<'\n';
                print_cromosome(30,results);
                std::cout<<"Size : "<<results.size()<<'\n';
                std::cout<<"best crom Size : "<<Best.size()<<'\n';
                print_cromosome(2,Best);
                std::cout<<"----------------------------------------------"<<'\n';
                file_name = "2.out";
                break;
            case 5:
                file_name = "5.out";
                break;
            case 10:
                file_name = "10.out";
                break;
            case 30:
                file_name = "30.out";
                break;
            case 100:
                file_name = "100.out";
                break;    
            default:
                file_name = "unknown.out";
                break;
        }
        std::cout << "Results for func_id: " << func_id << ", cromosome_size: " << cromosome_size << std::endl;
        std::cout << "Mean: " << mean << std::endl;
        std::cout << "Standard Deviation: " << stdev << std::endl;
        std::cout << "Global Minimum: " << min_val << std::endl;
        std::cout << "Results written to " << file_name << std::endl;

        // Deschidere fișier pentru scriere
        std::ofstream file(file_name, std::ios::app); // Append la fișier
        if (file.is_open()) {
            file << "Results for func_id: " << func_id << ", cromosome_size: " << cromosome_size << "\n";
            file << "Mean: " << (std::isnan(mean) || std::isinf(mean) ? "Invalid" : std::to_string(mean)) << "\n";
            file << "Standard Deviation: " << (std::isnan(stdev) || std::isinf(stdev) ? "Invalid" : std::to_string(stdev)) << "\n";
            file << "Global Minimum: " << (std::isnan(min_val) || std::isinf(min_val) ? "Invalid" : std::to_string(min_val)) << "\n\n";
            file.close();
        } else {
            std::cerr << "Error: Unable to open file " << file_name << std::endl;
        }

        std::cout << "Results written to " << file_name << std::endl;
    }
};

int main() {

//Hyper Parameeters
    int iterations = 1000;
    int pop_size = 100;
    double v_max = 0.5;
    int runs = 30;
    
    double personal_c = 2.0;
    double social_c = 1.9;

//for 2 dimension
    int dimensions = 2;
    std::string fileName = std::to_string(dimensions) + ".out";

    if (std::remove(fileName.c_str()) == 0) {
        std::cout << "Fișierul \"" << fileName << "\" a fost șters cu succes.\n";
    } else {
        std::perror("Eroare la ștergerea fișierului");
    }
    
    pop_size = 75;
    personal_c = 2.25;
    social_c = 2.25;
    iterations = 1700;
    v_max = 0.5;
    std::cout << "Optimizing Rastrigin function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(1,personal_c,social_c , iterations,pop_size, v_max, dimensions, runs);
    pop_size = 100;
    iterations = 1500;
    social_c = 2.0;
    personal_c = 2.0;
    v_max = 0.5;
    std::cout << "Optimizing Rosenbrock function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(2,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    //
    pop_size = 50;
    iterations = 1500;
    personal_c = 2.0;
    social_c = 2.0;
    v_max = 0.628;
    //
    std::cout << "Optimizing Michalewicz function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(3,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    pop_size = 100;
    iterations = 1500;
    personal_c = 2.0;
    social_c = 2.25;
    v_max = 0.5;
    std::cout << "Optimizing Griewangk function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(4,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);

//for 5 dimension
    dimensions = 5;
    fileName = std::to_string(dimensions) + ".out";

    if (std::remove(fileName.c_str()) == 0) {
        std::cout << "Fișierul \"" << fileName << "\" a fost șters cu succes.\n";
    } else {
        std::perror("Eroare la ștergerea fișierului");
    }
    
    pop_size = 75;
    personal_c = 2.25;
    social_c = 2.25;
    iterations = 1700;
    v_max = 0.5;
    std::cout << "Optimizing Rastrigin function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(1,personal_c,social_c , iterations,pop_size, v_max, dimensions, runs);
    pop_size = 100;
    iterations = 1500;
    social_c = 2.0;
    personal_c = 2.0;
    v_max = 0.5;
    std::cout << "Optimizing Rosenbrock function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(2,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    //
    pop_size = 100;
    iterations = 2300;
    personal_c = 2.25;
    social_c = 2.25;
    v_max = 0.5;
    //
    std::cout << "Optimizing Michalewicz function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(3,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    pop_size = 100;
    iterations = 1500;
    personal_c = 2.0;
    social_c = 2.25;
    v_max = 0.5;
    std::cout << "Optimizing Griewangk function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(4,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);

// for 10 dimensions
    dimensions = 10;

    fileName = std::to_string(dimensions) + ".out";

    if (std::remove(fileName.c_str()) == 0) {
        std::cout << "Fișierul \"" << fileName << "\" a fost șters cu succes.\n";
    } else {
        std::perror("Eroare la ștergerea fișierului");
    }

    personal_c = 2.25;
    social_c = 2.25;
    iterations = 1700;
    std::cout << "Optimizing Rastrigin function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(1,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    pop_size = 150;
    iterations = 1700;
    personal_c = 2.25;
    social_c = 1.70;
    std::cout << "Optimizing Rosenbrock function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(2,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    pop_size = 150;
    iterations = 2300;
    personal_c = 2.25;
    social_c = 2.25;
    v_max = 0.65;
    std::cout << "Optimizing Michalewicz function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(3,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    iterations = 2500;
    pop_size = 125;
    v_max = 0.65;
    personal_c = 2.25;
    social_c = 2.25;
    std::cout << "Optimizing Griewangk function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(4,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);

// for 30 dimensions
    dimensions = 30;

    fileName = std::to_string(dimensions) + ".out";

    if (std::remove(fileName.c_str()) == 0) {
        std::cout << "Fișierul \"" << fileName << "\" a fost șters cu succes.\n";
    } else {
        std::perror("Eroare la ștergerea fișierului");
    }

    iterations = 4000;
    pop_size = 225;
    v_max = 0.6;
    personal_c = 2.25;
    social_c = 2.25;
    std::cout << "Optimizing Rastrigin function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(1,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    pop_size = 150;
    iterations = 2300;
    personal_c = 2.25;
    social_c = 2.25;
    v_max = 0.65;
    std::cout << "Optimizing Rosenbrock function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(2,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    pop_size = 150;
    iterations = 2300;
    personal_c = 2.25;
    social_c = 2.25;
    v_max = 0.65;
    std::cout << "Optimizing Michalewicz function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(3,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    iterations = 4000;
    pop_size = 150;
    v_max = 0.7;
    personal_c = 2.25;
    social_c = 2.25;
    std::cout << "Optimizing Griewangk function for : " <<dimensions<<" dimensions"<< std::endl;
    Particle::particle_swarm_optimization(4,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);

// for 100 dimensions
    // dimensions = 100;

    // fileName = std::to_string(dimensions) + ".out";

    // if (std::remove(fileName.c_str()) == 0) {
    //     std::cout << "Fișierul \"" << fileName << "\" a fost șters cu succes.\n";
    // } else {
    //     std::perror("Eroare la ștergerea fișierului");
    // }

    // std::cout << "Optimizing Rastrigin function for : " <<dimensions<<" dimensions"<< std::endl;
    // Particle::particle_swarm_optimization(1,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    // std::cout << "Optimizing Rosenbrock function for : " <<dimensions<<" dimensions"<< std::endl;
    // Particle::particle_swarm_optimization(2,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    // std::cout << "Optimizing Michalewicz function for : " <<dimensions<<" dimensions"<< std::endl;
    // Particle::particle_swarm_optimization(3,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);
    // std::cout << "Optimizing Griewangk function for : " <<dimensions<<" dimensions"<< std::endl;
    // Particle::particle_swarm_optimization(4,personal_c,social_c , iterations, pop_size, v_max, dimensions, runs);

    return 0;
}
