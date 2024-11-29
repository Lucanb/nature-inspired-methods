import math
import random
import statistics
import time
from matplotlib import pylab, pyplot

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
        return x, y

def rastrigin(x):
    return 10 * len(x) + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

def griewangk(x):
    return sum(xi ** 2 / 4000 for xi in x) - math.prod(math.cos(xi / math.sqrt(i)) for i, xi in enumerate(x, 1)) + 1

def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))

def michalewicz(x):
    return -sum(math.sin(xi) * math.sin(i * xi ** 2 / math.pi) ** 20 for i, xi in enumerate(x, 1))

# FunctionMinimizer(rastrigin, 2, -5.12, 5.12).particle_swarm_optimization(1000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(griewangk, 2, -600, 600).particle_swarm_optimization(1000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(rosenbrock, 2, -2.048, 2.048).particle_swarm_optimization(5000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(michalewicz, 2, 0, math.pi).particle_swarm_optimization(1000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)

# FunctionMinimizer(rastrigin, 10, -5.12, 5.12).particle_swarm_optimization(5000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(griewangk, 10, -600, 600).particle_swarm_optimization(2000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(rosenbrock, 10, -2.048, 2.048).particle_swarm_optimization(10000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(michalewicz, 10, 0, math.pi).particle_swarm_optimization(5000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)

FunctionMinimizer(rastrigin, 30, -5.12, 5.12).particle_swarm_optimization(5000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)
FunctionMinimizer(griewangk, 30, -600, 600).particle_swarm_optimization(2000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)
FunctionMinimizer(rosenbrock, 30, -2.048, 2.048).particle_swarm_optimization(10000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)
FunctionMinimizer(michalewicz, 30, 0, math.pi).particle_swarm_optimization(5000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)

# FunctionMinimizer(rastrigin, 100, -5.12, 5.12).particle_swarm_optimization(10000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(griewangk, 100, -600, 600).particle_swarm_optimization(3000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(rosenbrock, 100, -2.048, 2.048).particle_swarm_optimization(10000, 100, 10, .729844, 1.496180, 1.496180, .01, .25)
# FunctionMinimizer(michalewicz, 100, 0, math.pi).particle_swarm_optimization(5000, 225, 15, .729844, 1.496180, 1.496180, .01, .25)