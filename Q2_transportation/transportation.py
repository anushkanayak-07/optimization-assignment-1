import numpy as np

cost = np.array([
    [4, 6, 8, 7],
    [5, 3, 7, 6],
    [6, 5, 4, 5]
], dtype=float)

supply = [30, 40, 20]
demand = [20, 30, 25, 15]


def vam(cost, supply, demand):
    supply = supply.copy()
    demand = demand.copy()
    allocation = np.zeros_like(cost)

    rows = set(range(len(supply)))
    cols = set(range(len(demand)))

    while rows and cols:
        row_penalty = {}
        col_penalty = {}

        for i in rows:
            x = sorted(cost[i, j] for j in cols)
            row_penalty[i] = x[0] if len(x) == 1 else x[1] - x[0]

        for j in cols:
            x = sorted(cost[i, j] for i in rows)
            col_penalty[j] = x[0] if len(x) == 1 else x[1] - x[0]

        r = max(row_penalty, key=row_penalty.get)
        c = max(col_penalty, key=col_penalty.get)

        if row_penalty[r] >= col_penalty[c]:
            i = r
            j = min(cols, key=lambda x: cost[i, x])
        else:
            j = c
            i = min(rows, key=lambda x: cost[x, j])

        amount = min(supply[i], demand[j])
        allocation[i, j] = amount

        supply[i] -= amount
        demand[j] -= amount

        if supply[i] == 0:
            rows.remove(i)

        if demand[j] == 0:
            cols.remove(j)

    return allocation


def find_path(basis, start, end, m, n):
    graph = {i: [] for i in range(m + n)}

    for i, j in basis:
        graph[i].append((m + j, (i, j)))
        graph[m + j].append((i, (i, j)))

    parent = {start: None}
    edge = {}
    stack = [start]

    while stack:
        node = stack.pop()

        if node == end:
            break

        for nxt, cell in graph[node]:
            if nxt not in parent:
                parent[nxt] = node
                edge[nxt] = cell
                stack.append(nxt)

    path = []
    node = end

    while node != start:
        path.append(edge[node])
        node = parent[node]

    return path[::-1]


def creates_cycle(basis, cell, m, n):
    graph = {i: [] for i in range(m + n)}

    for i, j in basis:
        graph[i].append(m + j)
        graph[m + j].append(i)

    start = cell[0]
    end = m + cell[1]

    stack = [start]
    visited = {start}

    while stack:
        node = stack.pop()

        if node == end:
            return True

        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)

    return False


def modi(cost, allocation, basis):
    m, n = cost.shape
    allocation = allocation.copy()
    basis = set(basis)

    while True:
        u = [None] * m
        v = [None] * n
        u[0] = 0

        changed = True

        while changed:
            changed = False

            for i, j in basis:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i, j] - u[i]
                    changed = True

                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i, j] - v[j]
                    changed = True

        delta = np.full((m, n), np.nan)

        for i in range(m):
            for j in range(n):
                if (i, j) not in basis:
                    delta[i, j] = cost[i, j] - u[i] - v[j]

        if np.nanmin(delta) >= 0:
            return allocation

        entering = np.unravel_index(np.nanargmin(delta), delta.shape)

        path = find_path(
            basis,
            entering[0],
            m + entering[1],
            m,
            n
        )

        cycle = [entering] + path

        minus = cycle[1::2]
        theta = min(allocation[i, j] for i, j in minus)

        for k, (i, j) in enumerate(cycle):
            if k % 2 == 0:
                allocation[i, j] += theta
            else:
                allocation[i, j] -= theta

        basis.add(entering)

        for cell in minus:
            if allocation[cell] == 0:
                basis.remove(cell)
                break


allocation = vam(cost, supply, demand)

basis = set(map(tuple, np.argwhere(allocation > 0)))

m, n = cost.shape

for i in range(m):
    for j in range(n):
        if len(basis) < m + n - 1:
            if (i, j) not in basis:
                if not creates_cycle(basis, (i, j), m, n):
                    basis.add((i, j))

print("Initial solution using VAM:")
print(allocation)

initial_cost = np.sum(allocation * cost)

print("\nInitial transportation cost:")
print(initial_cost)

optimal = modi(cost, allocation, basis)

print("\nOptimal allocation using MODI:")
print(optimal)

minimum_cost = np.sum(optimal * cost)

print("\nMinimum transportation cost:")
print(minimum_cost)
