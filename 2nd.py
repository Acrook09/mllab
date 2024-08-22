import numpy as np
def travellingsalesman(c):
    global cost
    adj_vertex = 999
    min_val = 999
    visited[c] = 1
    print((c + 1), end=" ")
    for k in range(n):
        if (tsp_g[c][k] != 0) and (visited[k] == 0):
            if tsp_g[c][k] < min_val:
                min_val = tsp_g[c][k]
                adj_vertex = k
    if min_val != 999:
        cost = cost + min_val
    if adj_vertex == 999:
        adj_vertex = 0
        print((adj_vertex + 1), end=" ")
        cost = cost + tsp_g[c][adj_vertex]
        return
    travellingsalesman(adj_vertex)
n = 4
cost = 0
visited = np.zeros(n, dtype=int)
tsp_g = np.array([[0, 10, 15, 20],
                  [10, 0, 35, 25],
                  [15, 35, 0, 30],
                  [20, 25, 30, 0]])
print("Shortest Path:", end=" ")
travellingsalesman(0)
print()
print("Minimum Cost:", end=" ")
print(cost)