import numpy as np
from typing import Dict
from GraphGenerator import generate_adjacency_matrix


def monte_carlo_simulation(
    n: int,
    expectedDegree: float,
    runs: int,
    rng: np.random.Generator
) -> Dict[str, float]:
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
    if n <= 0:
        raise ValueError("Number of vertices must be positive")

    if expectedDegree < 0:
        raise ValueError("Expected degree must be non-negative")

    if runs <= 0:
        raise ValueError("Number of runs must be positive")

    mean_degrees = []

    for _ in range(runs):
        adjacency_matrix = generate_adjacency_matrix(
            n=n,
            expectedDegree=expectedDegree,  # keep name consistent with your generator if needed
            rng=rng
        )

        degree_sequence = adjacency_matrix.sum(axis=1)
        mean_degrees.append(degree_sequence.mean())

    return {
        "expected_degree": float(expectedDegree),
        "empirical_mean_degree": float(np.mean(mean_degrees)),
        "empirical_std_degree": float(np.std(mean_degrees)),
        "runs": float(runs)
    }
