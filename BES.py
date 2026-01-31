import numpy as np
import time


def polr(a, R, N):
    th = a * np.pi * np.random.rand(N)
    r = th + R * np.random.rand(N)
    xR = r * np.sin(th)
    yR = r * np.cos(th)
    xR = xR / np.max(np.abs(xR))
    yR = yR / np.max(np.abs(yR))
    return xR, yR


def swoo_p(a, R, N):
    th = a * np.pi * np.exp(np.random.rand(N))
    r = th  # R * np.random.rand(N)
    xR = r * np.sinh(th)
    yR = r * np.cosh(th)
    xR = xR / np.max(np.abs(xR))
    yR = yR / np.max(np.abs(yR))
    return xR, yR


def BES(pop_pos, fobj, lb, ub, MaxIt):
    nPop, dim = pop_pos.shape
    # Initialize Best Solution
    BestSol = {'pos': None, 'cost': np.inf}
    convergence = np.zeros(MaxIt)
    low, high = lb[0], ub[0]
    ct = time.time()
    # Initialize population
    pop = {'pos': pop_pos, 'cost': np.zeros(nPop)}
    for i in range(nPop):
        pop['pos'][i, :] = low + (high - low) * np.random.rand(dim)
        pop['cost'][i] = fobj(pop['pos'][i, :])

        if pop['cost'][i] < BestSol['cost']:
            BestSol['pos'] = pop['pos'][i, :].copy()
            BestSol['cost'] = pop['cost'][i]

    # print(f"0 {BestSol['cost']}")

    # Main loop
    for t in range(1, MaxIt + 1):
        # Select space
        Mean = np.mean(pop['pos'], axis=0)
        lm = 2
        s1 = 0

        for i in range(nPop):
            newsol_pos = BestSol['pos'] + lm * np.random.rand(dim) * (Mean - pop['pos'][i, :])
            newsol_pos = np.maximum(newsol_pos, low)
            newsol_pos = np.minimum(newsol_pos, high)
            newsol_cost = fobj(newsol_pos)

            if newsol_cost < pop['cost'][i]:
                pop['pos'][i, :] = newsol_pos
                pop['cost'][i] = newsol_cost
                s1 += 1

                if pop['cost'][i] < BestSol['cost']:
                    BestSol['pos'] = pop['pos'][i, :].copy()
                    BestSol['cost'] = pop['cost'][i]

        # Search space
        a = 10
        R = 1.5
        s2 = 0

        for i in range(nPop - 1):
            A = np.random.permutation(nPop)
            pop['pos'] = pop['pos'][A, :]
            pop['cost'] = pop['cost'][A]

            x, y = polr(a, R, nPop)
            Step = pop['pos'][i, :] - pop['pos'][i + 1, :]
            Step1 = pop['pos'][i, :] - Mean
            newsol_pos = pop['pos'][i, :] + y[i] * Step + x[i] * Step1
            newsol_pos = np.maximum(newsol_pos, low)
            newsol_pos = np.minimum(newsol_pos, high)
            newsol_cost = fobj(newsol_pos)

            if newsol_cost < pop['cost'][i]:
                pop['pos'][i, :] = newsol_pos
                pop['cost'][i] = newsol_cost
                s2 += 1

                if pop['cost'][i] < BestSol['cost']:
                    BestSol['pos'] = pop['pos'][i, :].copy()
                    BestSol['cost'] = pop['cost'][i]

        # Swoop
        s3 = 0

        for i in range(nPop):
            A = np.random.permutation(nPop)
            pop['pos'] = pop['pos'][A, :]
            pop['cost'] = pop['cost'][A]

            x, y = swoo_p(a, R, nPop)
            Mean = np.mean(pop['pos'], axis=0)
            Step = pop['pos'][i, :] - 2 * Mean
            Step1 = pop['pos'][i, :] - 2 * BestSol['pos']
            newsol_pos = np.random.rand(dim) * BestSol['pos'] + x[i] * Step + y[i] * Step1
            newsol_pos = np.maximum(newsol_pos, low)
            newsol_pos = np.minimum(newsol_pos, high)
            newsol_cost = fobj(newsol_pos)

            if newsol_cost < pop['cost'][i]:
                pop['pos'][i, :] = newsol_pos
                pop['cost'][i] = newsol_cost
                s3 += 1

                if pop['cost'][i] < BestSol['cost']:
                    BestSol['pos'] = pop['pos'][i, :].copy()
                    BestSol['cost'] = pop['cost'][i]

        # Print best cost for the current iteration
        # print(f"{t} {BestSol['cost']}")
        convergence[t - 1] = BestSol['cost']
    Best_pos = BestSol['pos']
    best_fit = np.min(pop['cost'])
    ct = time.time() - ct
    return best_fit, convergence, Best_pos, ct

