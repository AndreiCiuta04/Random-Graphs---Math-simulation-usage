import numpy as np
from MonteCarloSimulation import monte_carlo_simulation


def main():
    """
    Entry point for the Monte Carlo simulation.

    @param n graph size
    @param runs number of Monte Carlo iterations
    @param seed random seed
    @param c_values expected degree values
    @return None
    """
    n = 10000
    runs = 100
    seed = 676767
    c_values = np.linspace(0.0, 2.0, 20)

    # Validating input
    if n <= 0:
        raise ValueError("n must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")
    for c in c_values:
        if c < 0 or c > n:
            raise ValueError("invalid expected degree")

    rng = np.random.default_rng(seed)

    for c in c_values:
        results = monte_carlo_simulation(n, c, runs, rng)
        print(
            f"c = {c:.2f} | "
            f"mean = {results['empirical_mean_degree']:.4f} | "
            f"std = {results['empirical_std_degree']:.4f}"
        )


if __name__ == "__main__":
    main()
