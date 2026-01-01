import matplotlib.pyplot as plt
import numpy as np


def plot_giant_component_zoom(results, c_min, c_max, title):
    """
    Plots the Monte Carlo estimate of E[|C_max|]/n as a function of c,
    restricted to the interval [c_min, c_max].

    @param results list of result dictionaries from monte_carlo_simulation
    @param c_min lower bound of c interval
    @param c_max upper bound of c interval
    @param title plot title
    @return None
    """
    filtered = [
        r for r in results
        if c_min <= r["expected_degree"] <= c_max
    ]

    c_values = [r["expected_degree"] for r in filtered]
    fractions = [r["mean_giant_component_fraction"] for r in filtered]

    plt.figure(figsize=(7, 5))
    plt.plot(c_values, fractions, marker="o", linestyle="-")
    plt.axvline(1.0, color="red", linestyle="--", label="Critical point c = 1")
    plt.xlabel("Average degree c")
    plt.ylabel("Fraction |C_max| / n")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_graph_degree_order(adj_list, title):
    """
    Visualizes a graph by ordering nodes by degree.

    Nodes are placed from left (highest degree) to right (degree 0).
    Vertical position is random jitter only.
    No geometric structure is imposed.

    @param adj_list adjacency list representation
    @param title plot title
    @return None
    """
    n = len(adj_list)

    # Compute degrees
    degrees = np.array([len(neigh) for neigh in adj_list])

    # Order nodes by decreasing degree
    order = np.argsort(-degrees)

    # Map old index -> new x-position
    x_pos = np.zeros(n)
    for i, u in enumerate(order):
        x_pos[u] = i

    # Small vertical jitter for readability
    rng = np.random.default_rng(42)
    y_pos = rng.normal(0.0, 0.3, size=n)

    plt.figure(figsize=(12, 4))

    # Draw edges
    for u in range(n):
        for v in adj_list[u]:
            if u < v:
                plt.plot(
                    [x_pos[u], x_pos[v]],
                    [y_pos[u], y_pos[v]],
                    color="black",
                    linewidth=0.6,
                    alpha=0.5
                )

    # Draw nodes
    plt.scatter(
        x_pos,
        y_pos,
        s=15,
        color="black",
        zorder=3
    )

    plt.title(title)
    plt.xlabel("Nodes ordered by decreasing degree")
    plt.yticks([])
    plt.tight_layout()
    plt.show()
