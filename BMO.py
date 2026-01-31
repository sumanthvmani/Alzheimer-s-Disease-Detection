import time
import numpy as np
import random


# Barnacles Mating Optimizer  (BMO) algorithm

def BMO(population, fobj, VRmin, VRmax, Max_iter):
    N, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    best_fitness = float('inf')
    best_solution = np.zeros((dim, 1))

    # Mating Optimizer parameters
    mutation_rate = 0.1
    crossover_rate = 0.8

    Convergence_curve = np.zeros((Max_iter))
    generation = 0
    ct = time.time()
    # Main optimization loop
    for generation in range(Max_iter):
        new_population = []

        for m in range(N // 2):
            # Select two parents using tournament selection
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            # Apply crossover with a certain probability
            if random.random() < crossover_rate:
                crossover_point = random.randint(0, 1)
                child1 = parent1 * crossover_point + parent2 * (1 - crossover_point)
                child2 = parent2 * crossover_point + parent1 * (1 - crossover_point)
            else:
                child1, child2 = parent1, parent2

            # Apply mutation with a certain probability
            if random.random() < mutation_rate:
                mutation = random.uniform(-0.1, 0.1)
                child1 += mutation
                child2 += mutation

            # Calculate the fitness for the children
            fitness1 = fobj(child1)
            fitness2 = fobj(child2)

            # Select the best child and add it to the new population
            best_child = child1 if fitness1 > fitness2 else child2
            new_population.append(best_child)

        # Replace the old population with the new one
        population = new_population

        # Find the best solution in the final population
        best_solution = max(population,  key=fobj)
        best_fitness = fobj(best_solution)

        Convergence_curve[generation] = best_fitness
        # generation = generation + 1
    best_fitness = Convergence_curve[Max_iter - 1]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
