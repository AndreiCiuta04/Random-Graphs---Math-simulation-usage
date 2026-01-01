import numpy as np
from MonteCarloSimulation import monte_carlo_simulation


def main():
    """
    Entry point for the Monte Carlo simulation.

    @param n graph size
    @param K number of Monte Carlo iterations
    @param seed random seed
    @param c_values expected degree values
    @return None
    """
    n = 10000
    K = 100
    seed = 676767
    c_values = np.linspace(0.0, 2.0, 20)

    # Validating input
    if n <= 0:
        raise ValueError("n must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    for c in c_values:
        if c < 0 or c > n:
            raise ValueError("invalid expected degree")

    rng = np.random.default_rng(seed)

    for c in c_values:
        results = monte_carlo_simulation(n, c, K, rng)
        print(
            f"c = {c:.2f} | "
            f"E[|C_max|]/n = {results['mean_giant_component_fraction']:.4f} | "
            f"std = {results['std_giant_component_fraction']:.4f}"
        )


if __name__ == "__main__":
    main()
