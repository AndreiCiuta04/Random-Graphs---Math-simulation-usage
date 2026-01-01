import numpy as np
from GraphGenerator import generate_adjacency_list


def monte_carlo_simulation(n, expectedDegree, K, rng):
    """
    Performs a Monte Carlo simulation for Erdős–Rényi random graphs.

    For fixed parameters (n, expectedDegree), the simulation repeatedly generates random
    graphs according to the G(n, p) model with p = c / n. The same random
    generator is reused across runs to ensure a reproducible sequence of
    independent samples.

    In each trial, the connected components of the graph are identified
    using a Breadth-First Search (BFS), and the size of the largest connected
    component |C_max| is saved in memeory.

    The results are averaged over all runs to estimate the expected fraction
    of vertices belonging to the giant component, |C_max| / n.

    @param n The number of vertices in each graph.
    @param c The average degree parameter, using p = c / n.
    @param K The number of Monte Carlo iterations.
    @param rng The random generator used for all simulations.
    @return A dictionary containing aggregated simulation results:
             - expectedDegree
             - mean_giant_fraction

    """

    proportions = []

    for _ in range(K):

        graph = generate_adjacency_list(n, expectedDegree, rng)
        c_max = largest_component_size(graph)
        proportions.append(c_max/n)

    return {
        "expected_degree": expectedDegree,
        "mean_giant_component_fraction": np.mean(proportions),
    }

def largest_component_size(graph):

    """
       Computes the size of the largest connected component using BFS.

       This implementation uses a Python list with an index pointer.

       @param graph Adjacency list representation of the graph.
       @return Size of the largest connected component.
       """

    n = len(graph)
    visited = [False] * n
    max_size = 0

    for start in range(n):
        #fixes a bug which caused the bfs to run over the same component over and over again
        if visited[start]:
            continue

        queue = [start]
        visited[start] = True
        head = 0
        size = 0

        while head < len(queue):
            i = queue[head]
            head += 1
            size += 1

            for v in graph[i]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        if size > max_size:
            max_size = size

    return max_size