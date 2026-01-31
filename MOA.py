import numpy as np
import time

# Fitness wrapper
def evaluate(fname, x):
    return fname(x)

def MOA(X, fname, xmin, xmax, Max_iter):
    # Masterpiece Optimization Algorithm
    start_time = time.time()

    # Initialization
    nPop, dim = X.shape
    fitness = np.array([evaluate(fname, x) for x in X])

    best_index = np.argmin(fitness)
    bestsol = np.copy(X[best_index])
    bestfit  = fitness[best_index]

    # Constants
    GF = 1
    cv2 = 0.1
    Z0 = 0.033
    kappa = 0.40

    for it in range(Max_iter):
        # Step 1
        r0 = (xmax - xmin) / 2
        r = r0 * (1 - it / Max_iter)

        # Equations (6)–(9) with separate a1, a2, a3, a4
        a1 = np.random.uniform(0, 2*np.pi)
        a2 = np.random.uniform(0, 2*np.pi)
        a3 = np.random.uniform(0, 2*np.pi)
        a4 = np.random.uniform(0, 2*np.pi)

        # Eq(6)
        x1 = bestsol + a1 * r
        x1 = np.clip(x1, xmin, xmax)

        # Eq(7)
        x2 = bestsol + a2 * r
        x2 = np.clip(x2, xmin, xmax)

        # Eq(8)
        x3 = bestsol + a3 * r
        x3 = np.clip(x3, xmin, xmax)

        # Eq(9)
        x4 = bestsol + a4 * r
        x4 = np.clip(x4, xmin, xmax)

        valley_points = np.vstack((x1, x2, x3, x4))
        valley_fits = np.array([evaluate(fname, x) for x in valley_points])

        # Combine with current population
        X_total = np.vstack((X, valley_points))
        fitness_total = np.hstack((fitness, valley_fits))

        # Select best nPop
        idx = np.argsort(fitness_total)
        X = X_total[idx[:nPop]]
        fitness = fitness_total[idx[:nPop]]

        # Update global best
        if fitness[0] < bestfit:
            bestfit = fitness[0]
            bestsol = np.copy(X[0])

        # Step 2
        A = np.random.uniform(0, 2*np.pi)
        yw = r + 1
        V = (1.0 / kappa) * np.log(yw / Z0)

        # Step 3
        Xnew = np.zeros_like(X)
        for i in range(nPop):
            rp = r * (1 + np.random.randn() * np.sqrt(cv2))
            xd_new = bestsol + rp * np.cos(A)
            yd_new = bestsol + rp * np.sin(A)
            dir_vec = np.sign(xd_new - X[i])
            Xnew[i] = X[i] + V * dir_vec
        Xnew = np.clip(Xnew, xmin, xmax)
        fitness_new = np.array([evaluate(fname, x) for x in Xnew])
        for i in range(nPop):
            if fitness_new[i] < fitness[i]:
                X[i] = Xnew[i]
                fitness[i] = fitness_new[i]
        if np.min(fitness) < bestfit:
            idx = np.argmin(fitness)
            bestfit = fitness[idx]
            bestsol = np.copy(X[idx])
    time3 = time.time() - start_time
    bestfit3 = bestfit
    fitness3 = fitness
    bestsol3 = bestsol
    return bestfit3, fitness3, bestsol3, time3
