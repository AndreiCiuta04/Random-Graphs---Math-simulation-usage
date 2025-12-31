import numpy as np
from GraphGenerator import generate_adjacency_matrix


def monte_carlo_simulation(n, expectedDegree, rng, runs):
    """
    Performs a Monte Carlo simulation for Erdős–Rényi random graphs.

    For fixed parameters (n, expected_degree), the simulation repeatedly
    generates random graphs and collects data. The same
    random generator is reused across runs to ensure a reproducible sequence
    of independent samples.

    The simulation currently measures the empirical mean degree across all
    vertices and runs.

    @param n: The number of vertices in each graph.
    @param expected_degree: The target expected degree parameter d, using p = d / n.
    @param runs: The number of Monte Carlo iterations.
    @param rng: The random generator used for all simulations.
    @return: A dictionary containing aggregated simulation results:
             - expected_degree
             - empirical_mean_degree
             - empirical_std_degree
             - runs
    @raises ValueError: If parameters are invalid.

    """

    mean_degrees = []

    for _ in range(runs):
        A = generate_adjacency_matrix(n, expectedDegree, rng)
        mean_degrees.append(A.sum(axis=1).mean())

    return {
        "expected_degree": expectedDegree,
        "empirical_mean_degree": np.mean(mean_degrees),
        "empirical_std_degree": np.std(mean_degrees),
        "runs": runs
    }
