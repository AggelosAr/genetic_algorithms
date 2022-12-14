import numpy as np
from numpy import random
import time
import multiprocessing
import concurrent.futures
from itertools import repeat
import matplotlib.pyplot as plt


# Fitness function based on the number of "1"s in the initial chromosome.
def fitness_function(a_chromosome):
    return np.count_nonzero(a_chromosome == 1)


# Decimal representation of a_chromosome.
def fitness_function_9_2(a_chromosome):
    return int(''.join([str(elem) for elem in a_chromosome]), 2)


# Creates a random chromosome of length l = 100.
def create_chromosome():
    return np.array([random.randint(2) for _ in range(100)])



def crossover(a_chromosome, b_chromosome, crossover_Rate = 0.7):
    if random.choice([0, 1],
                     p=[abs(1 - crossover_Rate), crossover_Rate]) == 1:  # The probability of crossover happening is X .
        p = random.choice(range(1, len(a_chromosome)))  # Point at which the replacement will take place.
        return np.hstack((a_chromosome[0:p], b_chromosome[p:])), np.hstack((b_chromosome[0:p], a_chromosome[p:]))



def mutation_point(a_chromosome, mutation_prop = 0.0001):
    # Scan the chromosome.
    for i in range(len(a_chromosome)):
        # Each point in the chromosome has a mutation probability of X.
        if random.choice([0, 1], p=[1 - mutation_prop, mutation_prop]) == 1:
            a_chromosome[i] = abs(a_chromosome[i] - 1)  # Flip bit
    return a_chromosome


def EVOLUTION(generation, crossover_Rate = 0.7, mutation_prob = 0.0001, return_fitness = False):
    # Calculate the fitness of the generation.
    
    fitness = np.array([[fitness_function(chromosome)] for chromosome in generation])

    # Roulette Wheel Selection. /start/
    parents = []
    # Create an array with probabilities based on the fitness level of each chromosome.
    prop = np.array([p / sum(fitness) for p in fitness])
    # Select top-k parents  based on their probability distribution. In pairs.
    for k in range(0, len(generation) // 2):
        choice = random.choice([i for i in range(len(generation))], p=prop.T[0], replace=False, size=2)
        parents.append(generation[choice])
    # Roulette Wheel Selection. /end/

    # This is where crossover happens and offsprings maybe produced.
    children = []

    for parent in parents:
        offsprings = crossover(parent[0], parent[1], crossover_Rate)
        if offsprings:
            children.append(offsprings)
        else:
            children.append(parent)  # clones.

   # Mutations.
    mutation_stage_children = []
    for child in children:
        mutation_stage_children.append(mutation_point(child[0], mutation_prob))
        mutation_stage_children.append(mutation_point(child[1], mutation_prob))

    mutation_stage_children = np.reshape(mutation_stage_children, (len(mutation_stage_children), generation.shape[1]))

    new_generation = np.concatenate((generation, mutation_stage_children), axis=0)

    fitness = np.array([[fitness_function(chromosome)] for chromosome in new_generation])
    # Sorts (from biggest to smallest) the fitness array and returns the INDEXES of the new_generation .
    top_fitness = np.flip(fitness.argsort(axis=0))
    # Select the top chromosomes from the new_generation.
    new_generation = np.array([new_generation[i] for i in top_fitness[0:(len(top_fitness) // 2)]]).reshape(
        new_generation.shape[0] // 2, new_generation.shape[1])

    if return_fitness:
        return new_generation, fitness
    return new_generation


def new_evolution(crossover_Rate = 0.7):
    population = 100
    optimal_chromosome = np.array([1 for _ in range(100)])

    total_stats = []

    for _ in range(20):

        start_time = time.time()
        generation = np.array([create_chromosome() for _ in range(population)])  # Creates the initial population.
        iteration = 0
        while True:

            generation = EVOLUTION(generation, crossover_Rate)
            # Check if stopping condition is met . If yes stop , else repeat.
            stop = False
            for element in generation:
                if np.array_equal(element, optimal_chromosome):
                    stop = True
                    break

            if stop:
                end_time = time.time()
                break
            else:
                iteration += 1

        total_stats.append([crossover_Rate, iteration, end_time - start_time])
    return total_stats




def new_evolution_9_2(population, crossover_Rate, mutation_prob):
    adaptation = []
    generation = np.array([create_chromosome() for _ in range(population)])
    iteration = 0
    while iteration < 100:
        
        generation, fitness = EVOLUTION(generation, crossover_Rate, mutation_prob, return_fitness = True)
        adaptation.append([np.max(fitness), np.mean(fitness), np.min(fitness)])
        iteration += 1

    return adaptation


def main9_1(params, cores):

    if isinstance(params, list):
        for p in params:
            if isinstance(p, int) or isinstance(p, float):
                pass
            else:
                print("Parameters error.")
                raise Exception
    else:
        print("Parameters error.")
        raise Exception

    if isinstance(cores, int) and multiprocessing.cpu_count() > cores > 0:
        print("Running with ", cores, "cores...")
    else:
        cores = multiprocessing.cpu_count()
        print("Running with ", cores, "cores...")
        input("Press any key to continue.")

    print(params)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        result = executor.map(new_evolution, params)

    stats = []
    for r in result:
        stats.append(r)

    stats = np.array(stats)

    return np.array(stats)


def main9_2(name, params, cores):

    names = ["crossover", "mutation", "population"]
    if name not in names:
        print("Name error.")
        raise Exception
    if isinstance(params, list):
        for p in params:
            if isinstance(p, int) or isinstance(p, float):
                pass
            else:
                print("Parameters error.")
                raise Exception
    else:
        print("Parameters error.")
        raise Exception

    if isinstance(cores, int) and multiprocessing.cpu_count() > cores > 0:
        print("Running with ", cores, "cores...")
    else:
        cores = multiprocessing.cpu_count()
        print("Running with ", cores, "cores...")
        #input("Press any key to continue.")


    if name == "population":
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            result = executor.map(new_evolution_9_2, params, repeat(0.7), repeat(0.0001))
    elif name == "crossover":
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            result = executor.map(new_evolution_9_2, repeat(100), params, repeat(0.0001))
    elif name == "mutation":
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            result = executor.map(new_evolution_9_2, repeat(100), repeat(0.7), params)


    stats = []
    for r in result:
        stats.append(r)

    a = np.array(stats)

    return stats
