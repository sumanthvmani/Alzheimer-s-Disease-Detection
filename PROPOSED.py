import time
import numpy as np
from math import gamma


def levy_flight(beta, dim):
    sigma = (
                    gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                    (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
            ) ** (1 / beta)

    u = np.random.normal(0, sigma, dim)  # Eq.(5)
    v = np.random.normal(0, 1, dim)  # Eq.(6)

    step = u / (np.abs(v) ** (1 / beta))  # Eq.(4)
    return step


def PROPOSED(positions, obj_func, lb, ub,  max_iter ):
    # Improved Flower Fertilization Optimization Algorithm
    # Proposed update is done at line 54
    agents, dim = positions.shape[0], positions.shape[1]

    gamma_init = 1.0
    beta_damp = 0.95
    levy_beta = 1.5
    velocities = positions.copy()
    fitness = np.array([obj_func(x) for x in positions])

    gamma_t = gamma_init
    best_costs = []

    ct = time.time()
    for t in range(max_iter):

        # Sort population
        idx = np.argsort(fitness)
        positions = positions[idx]
        velocities = velocities[idx]
        fitness = fitness[idx]

        X_best = positions[0]
        X_worst = positions[-1]
        X_middle = positions[len(positions) // 2]

        new_positions = []
        new_velocities = []
        new_fitness = []

        for i in range(agents):
            # Proposed update is done here
            # K = (X_best + X_middle + X_worst) / 3.0
            K = (np.min(fitness) + np.mean(fitness) + fitness[i] + np.max(fitness)) / 4.0

            L = levy_flight(levy_beta, dim)
            delta_S = L * (positions[i] - velocities[i])

            velocities[i] = velocities[i] * np.exp(-1 / gamma_t)

            rand_delta = np.random.rand(dim)
            new_x = (
                    positions[i]
                    - velocities[i]
                    + delta_S
                    - K * rand_delta
            )

            # Boundary control
            new_x = np.clip(new_x, lb[i], ub[i])

            new_positions.append(new_x)
            new_velocities.append(velocities[i])
            new_fitness.append(obj_func(new_x))

        positions = np.vstack((positions, np.array(new_positions)))
        velocities = np.vstack((velocities, np.array(new_velocities)))
        fitness = np.hstack((fitness, np.array(new_fitness)))

        idx = np.argsort(fitness)
        positions = positions[idx][:agents]
        velocities = velocities[idx][:agents]
        fitness = fitness[idx][:agents]
        best_costs.append(fitness[0])
        gamma_t *= beta_damp
    ct = time.time() - ct
    return  fitness, best_costs, positions, ct



