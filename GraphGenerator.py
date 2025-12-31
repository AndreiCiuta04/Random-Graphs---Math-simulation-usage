import numpy as np
from typing import Optional

"""
 Generates an undirected Erdős–Rényi random graph adjacency matrix.
 
  The graph follows the G(n, p) model with
      p = expectedDegree / n,
      E[deg] ≈ expectedDegree (for a sufficiently large n)
 
  The resulting adjacency matrix represents a simple graph, with the following proprieties:
   - undirected
   - no self-loops (meaning no v - v connection)
   - symmetric
   - allows only 0 or 1, as the each edge is unique
 
  Randomness will be controlled by providing an explicit random seed enabling a fully reproducible simulation and 
  Numpy's default random generator.
 
  @param n the number of vertices in the graph
  @param expectedDegree the target expected degree per vertex
  @param seed an optional seed for a custom number generator
  @param useDefaultRng if true it uses the Numpy random generator
  @return an n × n symmetric adjacency matrix
  @throws ValueError if parameters are invalid


"""
def generate_adjacency_matrix(
    n: int,
    expectedDegree: float,
    seed: Optional[int] = None,
    useDefaultRng: bool = False
) -> np.ndarray:
    p = expectedDegree / n

    if n <= 0:
        raise ValueError("Number of vertices must be positive")

    if expectedDegree < 0:
        raise ValueError("Expected degree must be non-negative")

    if p > 1:
        raise ValueError("Invalid parameters: p > 1")

    # Choose the source of random
    if useDefaultRng:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)


    upper_triangle = np.triu(random_matrix = rng.random((n, n)) < p, k=1)

    adjacency_matrix = upper_triangle + upper_triangle.T
    return adjacency_matrix.astype(int)
