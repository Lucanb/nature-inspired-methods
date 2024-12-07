import math
import random
import statistics
import time
from matplotlib import pylab, pyplot
import numpy as np

def circular_at(a, i):
    return a[i] if i < len(a) else a[i % len(a)]

def circular_neighborhood(a, i, k):
    return [circular_at(a, i - k // 2 + j) for j in range(k)]

class FunctionMinimizer:
    def __init__(self, f, n, xi_min, xi_max):
        self.f = f
        self.n = n
        self.xi_min = xi_min
        self.xi_max = xi_max

    def particle_swarm_optimization(self, steps, population_size, segment_length, w, phi_cog, phi_soc, eps, p):
        t1 = time.time()

        v_min = self.xi_min - self.xi_max
        v_max = self.xi_max - self.xi_min
        velocity = [tuple(random.uniform(v_min, v_max) for _ in range(self.n)) for _ in range(population_size)]

        position = [tuple(random.uniform(self.xi_min, self.xi_max) for _ in range(self.n)) for _ in range(population_size)]
        best_position = [*position]
        global_best_position = min(position, key=self.f)

        scores = [self.f(x) for x in position]
        stats = [(min(scores), max(scores), statistics.median(scores), statistics.stdev(scores))]

        for step in range(1, steps + 1):
            neighborhood_best_position = [min(circular_neighborhood(best_position, i, segment_length), key=self.f) for i in range(population_size)]
            velocity_norms = [sum(vi ** 2 for vi in v) for v in velocity]
            if statistics.median(velocity_norms) < eps:
                for i in range(population_size):
                    if random.random() < p:
                        velocity[i] = tuple(random.uniform(v_min, v_max) for _ in range(self.n))
                        position[i] = best_position[i] = tuple(random.uniform(self.xi_min, self.xi_max) for _ in range(self.n))

            for i in range(population_size):
                velocity_i = list(velocity[i])
                for d in range(self.n):
                    delta_cog = phi_cog * random.random() * (best_position[i][d] - position[i][d])
                    delta_soc = phi_soc * random.random() * (neighborhood_best_position[i][d] - position[i][d])
                    velocity_i[d] = max(min(w * velocity_i[d] + delta_cog + delta_soc, v_max), v_min)
                velocity[i] = tuple(velocity_i)

                position[i] = tuple(max(min(position[i][d] + velocity[i][d], self.xi_max), self.xi_min) for d in range(self.n))
                best_position[i] = min(best_position[i], position[i], key=self.f)
                global_best_position = min(global_best_position, position[i], key=self.f)

            if step % 100 == 0:
                print(step, self.f(global_best_position))
                scores = [self.f(x) for x in position]
                stats += [(min(scores), max(scores), statistics.median(scores), statistics.stdev(scores))]

        t2 = time.time()

        x, y = global_best_position, self.f(global_best_position)
        print(f"x = [{', '.join('{:.5f}'.format(xi) for xi in x)}]")
        print(f"y = {'{:.5f}'.format(y)}")
        print(f"TIME: {'{:.5f}'.format(t2 - t1)} seconds")
        print(self.f.__name__, self.n, stats[-1])

        metrics = ['min', 'max', 'mean', 'stdev']
        for i in range(4):
            pylab.gcf().canvas.manager.set_window_title(self.f.__name__ + '-' + str(self.n) + '-' + metrics[i])
            pyplot.plot(range(0, steps + 1, 100), [y[i] for y in stats])
            pyplot.show()

        return x, y, stats

def rastrigin(x):
    return 10 * len(x) + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

def griewangk(x):
    return sum(xi ** 2 / 4000 for xi in x) - math.prod(math.cos(xi / math.sqrt(i)) for i, xi in enumerate(x, 1)) + 1

def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))

def michalewicz(x):
    return -sum(math.sin(xi) * math.sin(i * xi ** 2 / math.pi) ** 20 for i, xi in enumerate(x, 1))


class GeneticAlgorithm:
    def __init__(self, param_ranges, population_size, generations, mutation_rate):
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        return [self.random_params() for _ in range(self.population_size)]

    def random_params(self):
        return {key: np.random.uniform(low, high) for key, (low, high) in self.param_ranges.items()}

    def mutate(self, params):
        for key in params:
            if np.random.rand() < self.mutation_rate:
                low, high = self.param_ranges[key]
                params[key] = np.random.uniform(low, high)
        return params

    def crossover(self, parent1, parent2):
        return {key: parent1[key] if np.random.rand() < 0.5 else parent2[key] for key in self.param_ranges}

    def evaluate_fitness(self, params, f, n, xi_min, xi_max):
        minimizer = FunctionMinimizer(f, n, xi_min, xi_max)
        _, best_value, _ = minimizer.particle_swarm_optimization(
            steps=1000,
            population_size=100,
            segment_length=10,
            w=params['w'],
            phi_cog=params['phi_cog'],
            phi_soc=params['phi_soc'],
            eps=params['eps'],
            p=params['p']
        )
        return best_value

    def run(self, f, n, xi_min, xi_max):
        population = self.initialize_population()
        for generation in range(self.generations):
            
            fitness_scores = [self.evaluate_fitness(individual, f, n, xi_min, xi_max) for individual in population]
            sorted_population = [individual for _, individual in sorted(zip(fitness_scores, population), key=lambda x: x[0])]
            
            top_individuals = sorted_population[:self.population_size // 2]
            new_population = top_individuals.copy()

            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(top_individuals, 2, replace=False)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)

            population = new_population

        return sorted_population[0]


def optimize_and_save(f, n, xi_min, xi_max, filename):
    param_ranges = {
        'w': (0.4, 0.9),
        'phi_cog': (0.5, 2.5),
        'phi_soc': (0.5, 2.5),
        'eps': (0.001, 0.1),
        'p': (0.1, 0.5)
    }

    ga = GeneticAlgorithm(param_ranges, population_size=10, generations=20, mutation_rate=0.1)
    best_params = ga.run(f, n, xi_min, xi_max)

    minimizer = FunctionMinimizer(f, n, xi_min, xi_max)
    _, best_value, stats = minimizer.particle_swarm_optimization(
        steps=1000,
        population_size=100,
        segment_length=10,
        w=best_params['w'],
        phi_cog=best_params['phi_cog'],
        phi_soc=best_params['phi_soc'],
        eps=best_params['eps'],
        p=best_params['p']
    )

    with open(filename, 'a') as file:
        file.write(f"Results for {f.__name__} ({n}D):\n")
        file.write(f"Best Parameters: {best_params}\n")
        file.write(f"Best Value: {best_value:.5f}\n")
        file.write(f"Stats (min, max, mean, stdev): {stats[-1]}\n")
        file.write("\n")


if __name__ == "__main__":
    functions = [rastrigin, griewangk, rosenbrock, michalewicz]
    dimensions = [2, 10, 30]

    for n in dimensions:
        filename = f"{n}d.txt"
        with open(filename, 'w') as file:
            file.write(f"Results for {n}-dimensional optimization:\n")
            file.write("\n")

        for f in functions:
            optimize_and_save(f, n, -5.12, 5.12, filename)

    print("Optimization complete. Results saved in files.")
