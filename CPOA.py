import time
import numpy as np


# Chaotic Puma Optimizer Algorithm (CPOA)

def CPOA(X, objective_function, VRmin, VRmax, max_iter):
    num_pelicans = X.shape[0]  # Population size
    num_variables = X.shape[1]  # Dimension
    lower_bound = VRmin[0, :]
    upper_bound = VRmax[0, :]

    fitness = np.array([objective_function(ind) for ind in X])
    best_index = np.argmin(fitness)
    X_best = X[best_index].copy()

    Convergence_curve = np.zeros((max_iter))
    ct = time.time()

    for t in range(max_iter):

        C = 2 - 2 * t / max_iter

        for i in range(num_pelicans):

            r1 = np.random.rand()
            r2 = np.random.rand()

            if r1 < 0.5:
                X_new = X[i] + C * r2 * (X_best - X[i])

            else:
                rand_index = np.random.randint(0, num_pelicans)
                X_rand = X[rand_index]
                X_new = X[i] + r2 * (X_rand - X[i])

            dive = np.random.uniform(-1, 1, num_variables)
            X_new = X_new + 0.1 * dive * (upper_bound - lower_bound)
            X_new = np.clip(X_new, lower_bound, upper_bound)

            new_fitness = objective_function(X_new)

            if new_fitness < fitness[i]:
                X[i] = X_new.copy()
                fitness[i] = new_fitness

                if new_fitness < objective_function(X_best):
                    X_best = X_new.copy()

        Convergence_curve[t] = objective_function(X_best)

    Leader_score = Convergence_curve[max_iter - 1]
    ct = time.time() - ct

    return Leader_score, Convergence_curve, X_best, ct
