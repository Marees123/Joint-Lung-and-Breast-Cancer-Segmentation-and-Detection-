import numpy as np
import time

# Educational Competition Optimizer (ECO)

def ECO(population, objective_function, lb, ub, max_iter):
    pop_size, dim = population.shape
    alpha = 2      # exploration control parameter
    beta = 1.5     # exploitation control parameter
    gamma = 0.8    # social factor

    best_solution = population[0, :].copy()
    best_fitness = objective_function(best_solution)
    convergence = np.zeros(max_iter)

    start_time = time.time()

    for iteration in range(max_iter):
        fitness_values = np.array([objective_function(ind) for ind in population])
        min_fitness_idx = np.argmin(fitness_values)
        min_fitness = fitness_values[min_fitness_idx]

        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[min_fitness_idx, :].copy()

        new_population = np.copy(population)

        T = 1 - iteration / max_iter

        for i in range(pop_size):
            dist = np.abs(population[i, :] - best_solution)

            rand1 = np.random.rand(dim)
            rand2 = np.random.rand(dim)

            move = (alpha * rand1 * (best_solution - population[i, :])) \
                   + (beta * rand2 * (dist * np.cos(2 * np.pi * rand2))) \
                   + gamma * T * (np.random.rand(dim) - 0.5)

            new_population[i, :] = population[i, :] + move
            new_population[i, :] = np.clip(new_population[i, :], lb[i, :], ub[i, :])

        population = new_population


        fitness_values = np.array([objective_function(ind) for ind in population])
        min_fitness_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[min_fitness_idx]
        best_solution = population[min_fitness_idx, :].copy()

        convergence[iteration] = best_fitness

    computation_time = time.time() - start_time
    return best_fitness, convergence, best_solution, computation_time