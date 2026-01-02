import numpy as np
from MonteCarloSimulation import monte_carlo_simulation
from GraphGenerator import generate_adjacency_list
from Visualization import plot_giant_component_zoom, plot_graph_degree_order


def main():
    """
    Entry point for the Monte Carlo simulation.

    @param n graph size
    @param runs number of Monte Carlo iterations
    @param seed random seed
    @param c_values expected degree values
    @return None
    """
    # Monte Carlo parameters (paper)
    n = 10000
    runs = 100
    seed = 676767
    c_values = np.linspace(0.0, 2.0, 20)

    rng = np.random.default_rng(seed)

    results_list = []

    for c in c_values:
        results = monte_carlo_simulation(n, c, runs, rng)
        results_list.append(results)
        print(
            f"c = {c:.2f} | "
            f"E[|C_max|]/n = {results['mean_giant_component_fraction']:.4f}"
        )

    # Quantitative plots (paper)
    plot_giant_component_zoom(
        results_list,
        c_min=0.0,
        c_max=2.0,
        title="Largest component size as a function of c"
    )

    # Illustrative visualization (small n, not part of simulation)
    n_vis = 500
    c_vis = 0.3
    graph = generate_adjacency_list(n_vis, c_vis, rng)
    plot_graph_degree_order(
        graph,
        f"Degree-ordered visualization (n={n_vis}, c={c_vis})")

    c_vis = 0.67
    graph = generate_adjacency_list(n_vis, c_vis, rng)
    plot_graph_degree_order(
        graph,
        f"Degree-ordered visualization (n={n_vis}, c={c_vis})")

    c_vis = 0.9
    graph = generate_adjacency_list(n_vis, c_vis, rng)
    plot_graph_degree_order(
        graph,
        f"Degree-ordered visualization (n={n_vis}, c={c_vis})")

    c_vis = 1
    graph = generate_adjacency_list(n_vis, c_vis, rng)
    plot_graph_degree_order(
        graph,
        f"Degree-ordered visualization (n={n_vis}, c={c_vis})")

    c_vis = 1.2
    graph = generate_adjacency_list(n_vis, c_vis, rng)
    plot_graph_degree_order(
        graph,
        f"Degree-ordered visualization (n={n_vis}, c={c_vis})")

    c_vis = 1.5
    graph = generate_adjacency_list(n_vis, c_vis, rng)
    plot_graph_degree_order(
        graph,
        f"Degree-ordered visualization (n={n_vis}, c={c_vis})")

    c_vis = 2
    graph = generate_adjacency_list(n_vis, c_vis, rng)
    plot_graph_degree_order(
        graph,
        f"Degree-ordered visualization (n={n_vis}, c={c_vis})"
    )

if __name__ == "__main__":
   main()
