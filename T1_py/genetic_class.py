import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from ttp import TTP
import json

LIMIT_SOLUTION = {
    'a280-n279': 100,
    'a280-n1395': 100,
    'a280_n2790': 100,
    'fnl4461-n4460': 50,
    'fnl4461-n22300': 50,
    'fnl4461-n44600': 50,
    'pla33810-n33809': 20,
    'pla33810-n169045': 20,
    'pla33810-n338090': 20
}

class GA:
    def __init__(self,file_name) -> None:
        self.population = []    #Atentie cu nr-ul de obiective ale functiei
        self.pop_size = 0
        self.n_objectives = 2
        self.distances = []
        self.file_name = file_name
        self.profits = []
        self.weights = []
        self.items = []

        content = []
        print(file_name)
        test_name = open(f"test_problems/{file_name}.txt")
        for i in test_name:
            content.append(i.split())
        
        #aici trebuie atentie la ce coloane alegem
        self.city_number = int(content[2][-1])    # total number of cities
        self.capacity = int(content[4][-1])  # threshold value
        self.minimum_speed = float(content[5][-1])        # minimum speed
        self.maximum_speed = float(content[6][-1])       # maximum speed
        self.renting_ratio = float(content[7][-1]) # renting ratio

        del content[0:10]   #datele astea nu ma mai intereseaza

        #acum primul lucru pe care il facem ca sa exploatam fisierul este sa luam in parte fiecare pereche de drumuri
        nodes = []
        for i in range(self.city_number):
            nodes.append([eval(j) for j in content[i]])
        del content[0:self.city_number+1]

        self.distances = self.calculateDistance(nodes)

        for line in content :
            self.profits.append(int(line[1])) # profit value      
            self.weights.append(int(line[2])) # weight value
            self.items.append(int(line[3]))   # node assigned
        
        #Let's zipp for faster coding + sorting

        zipVec = zip(self.items,self.profits,self.weights)
        zip_vec = sorted(zipVec)
        self.items,self.profits,self.weights = zip(*zip_vec)


    def population_generator(self,length):
        self.length = length
        population = []
        
        number_items = len(self.items)
        items_chosen = [random.choice([0,1]) for k in range(number_items)]
        route = [range(1, self.city_number+1), self.city_number]

        population.append(TTP(
            self.distances,
            self.capacity,
            self.minimum_speed,
            self.maximum_speed,
            self.profits,
            self.weights,
            self.items,
            route,
            items_chosen
        ))

        self.population = np.array(population)

    def calculateDistance(self,nodes):
        
        node_coords = np.array(nodes).reshape(-1,3)[:,1:] # convert node list into numpy array of x and y coords
        distances = np.sqrt(np.sum((node_coords[:, np.newaxis] - node_coords) ** 2, axis=-1)) # create distance matrix from coords
        return distances
    
    def crossover_TwoPoints(self,parent1=[],parent2=[]):
        
        parent1_copy = copy.deepcopy(parent1)
        parent2_copy = copy.deepcopy(parent2)

        #generam 2 pct cross-over random

        cross_point1 = random.randint(0,self.city_number-1)
        cross_point2 = random.randint(0,self.city_number-1)
        while cross_point1 == cross_point2:
            cross_point1 = random.randint(0,self.city_number-1)
        if cross_point1 > cross_point2 :
            temp = cross_point2
            cross_point2 = cross_point1
            cross_point1 = temp
        
        middle_part1 = parent1_copy[cross_point1:cross_point2]
        middle_part2 = parent2_copy[cross_point1:cross_point2]
        # head_part1 = parent1[:cross_point1]
        # head_part2 = parent2[:cross_point1]
        # tail_part1 = parent1[cross_point2:]
        # tail_part2 = parent2[cross_point2:]

# e destul sa schimbam numai mijloacele
        parent1_copy_new = []
        for i in parent1_copy[:cross_point1]:
            while i in middle_part2: 
                i = middle_part1[middle_part2.index(i)] 
            parent1_copy_new.append(i)
        parent1_copy_tail = []
        for i in parent1_copy[cross_point2:]:
            while i in middle_part2:
                i = middle_part1[middle_part2.index(i)]
            parent1_copy_tail.append(i)
        parent1_final = parent1_copy_new + middle_part2 + parent1_copy_tail #set the crossover part untouched and add fixed head part and tail part

        parent2_copy_new = []
        for i in parent2_copy[:cross_point1]: 
            while i in middle_part1: 
                i = middle_part2[middle_part1.index(i)]
            parent2_copy_new.append(i)
        parent2_copy_tail = []
        for i in parent2_copy[cross_point2:]:
            while i in middle_part1:
                i = middle_part2[middle_part1.index(i)]
            parent2_copy_tail.append(i)
        parent2_final = parent2_copy_new + middle_part1 + parent2_copy_tail

        children1 = copy.deepcopy(parent1_final)
        children2 = copy.deepcopy(parent2_final)

        return children1, children2
    
    def orderedCrossOver(self,parent1=[],parent2=[]):
        
        parent1_copy = copy.deepcopy(parent1)
        parent2_copy = copy.deepcopy(parent2) 

        order_point1 = random.randint(0, self.city_number-1)
        order_point2 = random.randint(0, self.city_number-1)

        while order_point2 == order_point1:
            order_point2 = random.randint(0, self.city_number-1)
        if order_point1 > order_point2:
            temp = order_point1
            order_point1 = order_point2
            order_point2 = temp
        
        parent1_head = [None]*order_point1
        parent1_tail = [None]*(self.city_number - order_point2)
        middle1 = parent1_copy[order_point1:order_point2]
        parent1_copy_overiting = parent1_head + middle1 + parent1_tail

        parent2_head = [None]*order_point1
        parent2_tail = [None]*(self.city_number - order_point2)
        middle2 = parent2_copy[order_point1:order_point2]
        parent2_copy_overiting = parent2_head + middle2 + parent2_tail

        p1_remain = [i for i in parent2 if i not in parent1_copy_overiting]
        parent1_copy_overiting[:order_point1] = p1_remain[:order_point1]
        parent1_copy_overiting[order_point2:] = p1_remain[order_point1:]

        p2_remain = [i for i in parent1 if i not in parent2_copy_overiting]
        parent2_copy_overiting[:order_point1] = p2_remain[:order_point1]
        parent2_copy_overiting[order_point2:] = p2_remain[order_point1:]

        children1 = copy.deepcopy(parent1_copy_overiting)
        children2 = copy.deepcopy(parent2_copy_overiting)
        return children1, children2
    
    def kp_crossover(self,parent1,parent2):
        
        parent1_copy = copy.deepcopy(parent1)
        parent2_copy = copy.deepcopy(parent2)
        split_point = np.random.randint(0,len(parent1_copy))
        child1 = parent1_copy[:split_point] + parent2_copy[split_point:]
        child2 = parent2_copy[:split_point] + parent1_copy[split_point:]

        return child1,child2

    def inversionMutation (self, parent1 = [], parent2 = []):
        parent1_copy = copy.deepcopy(parent1)
        parent2_copy = copy.deepcopy(parent2)
        inverse_split1 = random.randint(0, self.city_number-1)
        inverse_split2 = random.randint(0, self.city_number-1)

        parent1_head = parent1_copy[:inverse_split1]
        parent1_tail = parent1_copy[inverse_split1:]
        
        parent1_tail.reverse()
        parent2_head = parent2_copy[:inverse_split2]
        parent2_tail = parent2_copy[inverse_split2:]
        parent2_tail.reverse()

        parent1_new = parent1_tail + parent1_head
        parent2_new = parent2_tail + parent2_head
        children1 = copy.deepcopy(parent1_new)
        children2 = copy.deepcopy(parent2_new)
        return children1, children2
    
    def kpMutation(self,parent=[]):
        point1, point2 = sorted(random.sample(range(len(parent)), 2))
        middleSequence = parent[point1:point2 + 1]
        parent[point1:point2 + 1] = middleSequence[::-1]
        return parent

#folosim asta pentru a depista punctele Pareto ----------------------------------------------------->

    def nonDominatedSorting(self):

        dominating_sets = []
        dominated_counts = []

        for solution_1 in self.population:
            current_dominating_set = set()
            dominated_counts.append(0)
            for i, solution_2 in enumerate(self.population):
                if solution_1 >= solution_2 and not solution_1 == solution_2:
                    current_dominating_set.add(i)
                elif solution_2 >= solution_1 and not solution_2 == solution_1:
                    dominated_counts[-1] += 1
            dominating_sets.append(current_dominating_set)

        dominated_counts = np.array(dominated_counts)
        self.fronts = []

        while True:
            current_front = np.where(dominated_counts==0)[0]
            if len(current_front) == 0:
                break
            self.fronts.append(current_front)
            for individual in current_front:
                dominated_counts[individual] = -1
                dominated_by_current_set = dominating_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    dominated_counts[dominated_by_current] -= 1

    def calc_crowding_distance(self):
        self.crowding_distance = np.zeros(len(self.population))

        for front in self.fronts:
            fitnesses = np.array([
                solution.get_fitness() for solution in self.population[front]
            ])
        
            # Normalise each objectives, so they are in the range [0,1]
            # This is necessary, so each objective's contribution have the same magnitude to the crowding distance.
            normalized_fitnesses = np.zeros_like(fitnesses)

            for j in range(self.n_objectives):
                min_val = np.min(fitnesses[:, j])
                max_val = np.max(fitnesses[:, j])
                val_range = max_val - min_val
                normalized_fitnesses[:, j] = (fitnesses[:, j] - min_val) / val_range

            for j in range(self.n_objectives):
                idx = np.argsort(fitnesses[:, j])
                
                self.crowding_distance[idx[0]] = np.inf
                self.crowding_distance[idx[-1]] = np.inf
                if len(idx) > 2:
                    for i in range(1, len(idx) - 1):
                        self.crowding_distance[idx[i]] += normalized_fitnesses[idx[i + 1], j] - normalized_fitnesses[idx[i - 1], j]
        
        
         
    # Visualisation function
    def visualize(self):
        for front in self.fronts:
            pareto_value = np.array([solution.get_fitness() for solution in self.population[front]])
            plt.scatter(
                pareto_value[:, 0],
                pareto_value[:, 1],
            )
        plt.xlabel('travelling time')
        plt.ylabel('total profit')
        plt.grid()
        plt.show()
        

    def calculate_Z(self,total_profit,travelling_time): ####

        profit_sum = total_profit
        travel_time = travelling_time
        rent_cost = self.renting_ratio * travel_time
        Z_value = profit_sum - rent_cost
        return Z_value

    # Results export
    def export_result(self):
        DIR = 'test_results/'
        with open(f'{DIR}/TeamU_{self.file_name}.f','w') as f:
            count = 0
            for solution in self.population[self.fronts[0]]:
                f.write(f"{solution.travel_time} {solution.final_profit}\n")
                count += 1
                print(solution.travel_time)
                print(solution.final_profit)
                Z   = self.calculate_Z(solution.final_profit,solution.travel_time) ####
                print(Z)
                if count == LIMIT_SOLUTION[self.file_name]:
                    break

        with open(f'{DIR}/TeamU_{self.file_name}.x','w') as f:
            count = 0
            for solution in self.population[self.fronts[0]]:
                f.write(f"{str(solution.routes)[1:-1].replace(',', '')}\n")
                f.write(f"{str(solution.items)[1:-1].replace(',', '')}\n")
                f.write('\n')
                count += 1
                if count == LIMIT_SOLUTION[self.file_name]:
                    break
    
    
    # Elitism Replacement function
    def elitism_replacement(self):
        elitism = copy.deepcopy(self.population)
        population = []
        
        i = 0
        while len(self.fronts[i]) + len(population) <= self.pop_size:
            for solution in elitism[self.fronts[i]]:
                population.append(solution)
            i += 1

        front = self.fronts[i]
        ranking_index = front[np.argsort(self.crowding_distance[front])]
        current_pop_len = len(population)
        for index in ranking_index[current_pop_len:self.pop_size]:
            population.append(elitism[index])
        self.population = np.array(population)

    
    # Tournament selection function
    def tournament_selection(self):
        tournament = np.array([True] * self.size_t + [False] * (self.pop_size - self.size_t))
        results = []
        for _ in range(2):
            np.random.shuffle(tournament)
            front = []
            for f in self.fronts:
                front = []
                for index in f:
                    if tournament[index] == 1:
                        front.append(index)
                if len(front) > 0:
                    break
            max_index = np.argmax(self.crowding_distance[front])
            results.append(self.population[front[max_index]])
        return results


    # This actually 'runs' our GA
    def optimize(self, generations, tournament_size, crossover='OX', selection = '', mutation = '', replacement = ''):
        self.size_t = tournament_size

        for generation in range(generations):
            print('Generation: ', generation + 1)
            new_solutions = []
            self.nonDominatedSorting()
            self.calc_crowding_distance()
            while len(self.population) + len(new_solutions) < 2 * self.pop_size:
                
                if selection == 'roulette':
                    parents = self.roulette_wheel_selection()
                else:
                    parents = self.tournament_selection()
                    
                parents = self.tournament_selection()
                
                if crossover == 'PMX':
                    route_child_a, route_child_b = self.crossover_TwoPoints(parents[0].route, parents[1].route)
                else:
                    route_child_a, route_child_b = self.orderedCrossOver(parents[0].route, parents[1].route)   
                stolen_child_a, stolen_child_b = self.kp_crossover(parents[0].stolen_items, parents[1].stolen_items)
                
                if mutation == 'insertion':
                    new_route_c, new_route_d = self.tsp_insertion_mutation(route_child_a), self.tsp_insertion_mutation(route_child_b)
                else: 
                    new_route_c, new_route_d = self.inversionMutation(route_child_a, route_child_b)
                new_stolen_c = self.kpMutation(stolen_child_a) 
                new_stolen_d = self.kpMutation(stolen_child_b)
                    
                
                new_solutions.append(
                    TTP(
                        self.distances,
                        self.capacity,
                        self.minimum_speed,
                        self.maximum_speed,
                        self.profits,
                        self.weights,
                        self.items,
                        new_route_c,
                        new_stolen_c,
                        self.renting_ratio
                    )
                )
                new_solutions.append(
                    TTP(
                        self.distances,
                        self.capacity,
                        self.minimum_speed,
                        self.maximum_speed,
                        self.profits,
                        self.weights,
                        self.items,
                        new_route_d,
                        new_stolen_d,
                        self.renting_ratio
                    )
                )

            self.population = np.append(self.population, new_solutions)
            self.nonDominatedSorting()
            self.calc_crowding_distance()
            
            if replacement == 'non-elitist':
                new_solutions = self.non_elitist_replacement()
            else: 
                self.elitism_replacement()
            
        self.nonDominatedSorting()
        self.calc_crowding_distance()
    

        
    # Evaluation function
    def evaluate_solution(solution, weight_list, profit_list, knapsack_capacity):
        """
        Evaluates a solution to the knapsack problem, calculating its total profit and weight.
    
        :param solution: List representing the solution (1 if item is included, 0 otherwise).
        :param weight_list: List of weights of the items.
        :param profit_list: List of profits of the items.
        :param knapsack_capacity: Maximum allowable weight in the knapsack.
        :return: Tuple (total profit, total weight) of the solution. If the total weight exceeds
                 the capacity, the profit is set to 0.
        """
        total_weight = sum(solution[i] * weight_list[i] for i in range(len(solution)))
        total_profit = sum(solution[i] * profit_list[i] for i in range(len(solution)))
        if total_weight > knapsack_capacity:
            total_profit = 0  
        return total_profit, total_weight
    
    # Yields neighbouring solutions to current solution
    def get_neighbor(current_solution):
        """
        Generator that yields all the neighboring solutions of the current solution.
    
        A neighboring solution is generated by flipping one item's inclusion status
        (from 0 to 1 or from 1 to 0) in the solution.
    
        :param current_solution: List representing the current solution.
        :yield: A neighboring solution.
        """
        for i in range(len(current_solution)):
            neighbor = current_solution[:]
            neighbor[i] = 1 - neighbor[i]
            yield neighbor
    
    # Local search function
    def local_search(weight_list, profit_list, knapsack_capacity, max_iter=10):
        """
        Performs local search to find an optimal or near-optimal solution to the knapsack problem.
    
        The algorithm starts with a random solution and iteratively moves to neighboring solutions
        if they provide a higher profit, until no improvement is found or the maximum iterations are reached.
    
        :param weight_list: List of weights of the items.
        :param profit_list: List of profits of the items.
        :param knapsack_capacity: Maximum allowable weight in the knapsack.
        :param max_iter: Maximum number of iterations for the local search.
        :return: Tuple (best solution, best solution value).
        """
        # Generate an initial random solution within the knapsack capacity
        current_solution = [random.choice([0,1]) for _ in range(len(weight_list))]# Random generation of initial solutions
        # Make sure this solution is not overweight
        while sum(current_solution[i] * weight_list[i] for i in range(len(current_solution))) > knapsack_capacity:
            current_solution = [random.choice([0,1]) for _ in range(len(weight_list))]
        current_solution = list(current_solution)
        
        # Calculate the current solution value and weight
        current_solution_value, current_solution_weight = evaluate_solution(current_solution, weight_list, profit_list, knapsack_capacity)
        #copy the current solution to best solution for further compare
        best_solution = current_solution.copy()
        best_solution_value = current_solution_value
        
        # Do the local search
        for j in range(max_iter):
            print('iteration: ' , j)
            found_better = False
            for neighbor_solution in get_neighbor(current_solution):
                neighbor_solution_value = evaluate_solution(neighbor_solution, weight_list, profit_list, knapsack_capacity)[0]
                
                if neighbor_solution_value > best_solution_value:
                    best_solution = neighbor_solution[:]
                    best_solution_value = neighbor_solution_value
                    found_better = True
    
            if not found_better:
                break
    
            current_solution = best_solution[:]
            print(current_solution)
            current_solution_value = best_solution_value
            print(current_solution_value)
        return best_solution, best_solution_value
    
    
    
    
    
    def roulette_wheel_selection(self):
        # Sums up the total fitness for all solutions
        total_fitness = np.sum([solution.get_fitness() for solution in self.population])
        # Finds the probability of each solution by their fitness
        probabilities = [solution.get_fitness() / total_fitness for solution in self.population]
    
        # Cumulative probabilities
        cumulative_probabilities = np.cumsum(probabilities)
    
        # Select indices using roulette wheel selection
        selected_indices = []
        for _ in range(self.size_t):
            random_number = np.random.rand()
            selected_index = np.searchsorted(cumulative_probabilities, random_number)
            selected_indices.append(selected_index % len(self.population))
        return [self.population[i] for i in selected_indices]

    def tsp_insertion_mutation(self, parent1=[]):
        # copies parent1
        p1 = copy.deepcopy(parent1)
        # defines random mutation point
        mutation_point = random.randint(0, self.city_number - 1)
        # defines new point
        new_position = random.randint(0, self.city_number - 1)
        # removes mutation point and reinserts it at new point
        p1.pop(mutation_point)
        p1.insert(new_position, mutation_point + 1)
        return p1
    
    def non_elitist_replacement(self):
        # randomly selects induviduals
        selected_indices = np.random.choice(len(self.population), size=self.pop_size, replace=False)
        return [self.population[i] for i in selected_indices]