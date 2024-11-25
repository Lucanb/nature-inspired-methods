import numpy as np

class TTP:

    def __init__(self, distances, capacities,minimum_speed,maximum_speed,profits,weights,items,routes,stolens):
        self.distance = distances
        self.capacities = capacities
        self.minimum_speed = minimum_speed
        self.maximum_speed = maximum_speed
        self.profits = profits
        self.weights = weights
        self.items = items
        self.routes = routes
        self.stolens = stolens

        self.num_cities = len(self.distance)

        self.travel_time = self.get_travelTime()
        self.final_profit = self.fitness_profit()

# o probema vad  - la comparatie trebuie ca noi sa facem o formula care scade din profit timpul pentru avedea exact deoarece acesta are o rata
    def __gt__(self,other):
        return (self.travel_time < other.travel_time) and \
        (self.final_profit > other.final_profit)
    
    def __eq__(self, other):
        return (self.travel_time == other.travel_time) and \
        (self.final_profit == other.final_profit)

    def __ge__(self,other):
        return (self.travel_time >= other.travel_time) and \
        (self.final_profit <= other.final_profit)

# -------------

    def get_weight(self,index):
        stolenItem = self.stolens
        route = self.routes
        weight = self.weights
        sum_weght = 0
        for index in range(len(route)):
            item_in_city = np.where(self.items == route[index])[0]
            for j in item_in_city:
                if stolenItem[j]:
                    sum_weght += weight[j]
        return sum_weght
    
    def get_velocity(self,index):

        weight = self.get_weight(index)
        capacity_constant = weight/self.capacities
        velocity = capacity_constant*np.abs((self.maximum_speed - self.minimum_speed)) #formula velocity vmax - Wc(max_speed - min_speed/W)
        if capacity_constant <= self.capacities: # de verificat cum alegem capacitatea (ambele sunt rapoarte)
            velocity = self.maximum_speed - velocity # If weight<=capacity return reduced velocity
        else:
            velocity = self.minimum_speed # o solutie invalida asa ca vom incerca pentru moment sa ne intoarcem unde eram (ar merge sa penalizam chiar)
        return velocity
    
    def get_travelTime(self):
        totalTime = 0 # initializare cu 0
        routeWeight = 0
        route = self.routes # avem ruta ca sa luam distantele

        for i in range(len(route) - 1): # acum mergem pe perechi de orase consecutive din ruta
            distanceDone = self.distance[i][i + 1] #distanta parcursa
            accumulatedWeight = self.get_weight(i) #greutatea pe care o avem pana acum
            if accumulatedWeight > self.capacities: #verificam capacitatea
                routeWeight = -float('inf') # clar am depasit avem solutie invalida si trebuie ceva mare ca sa eliminam la fitness - unul cu - si altul cu +
                totalTime = float('inf')
                break
            velocituGet = self.get_velocity(i) # viteza este egala din formula velocitatii
            totalTime += distanceDone / velocituGet # timpul este egal cu distanta parcursa / viteza pe care o avem(velocity)
            routeWeight += accumulatedWeight #simplu adunam ca sa facem timpul total
        totalTime += self.distance[len(route) - 1][0] / self.get_velocity(
            len(route) - 1) # aici el trebuie sa se intoarca deci ciobaneste adunam la final :))
        return totalTime

    def fitness_profit(self):
        profit = 0
        for item,stolen in enumerate(self.items):
            if stolen:
                profit += self.profits[item]
        return profit

    def get_fitness(self):
        return np.array([self.travel_time, self.final_profit])

    def best_improve_optimizer(self,best_improvement = False):
        new_route = self.routes.copy()
        route = self.routes
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:] 
                    new_route[i:j] = route[j - 1:i - 1:-1] 
                    if self.get_travelTime(new_route) < self.get_travelTime(route):
                        route = new_route
                        if not best_improvement:
                            self.route = route
                            return

        self.routes = route
    



