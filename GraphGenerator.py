import numpy as np

"""
 Generates an undirected Erdős–Rényi random graph adjacency list.
 
  The graph follows the G(n, p) model with
      p = expectedDegree / n,
      E[deg] ≈ expectedDegree (for a sufficiently large n)
 
  The resulting adjacency list represents a simple graph, with the following proprieties:
   - undirected
   - no self-loops (meaning no v - v connection)
   - symmetric
   - each edge is unique
 
  Randomness will be controlled by providing an explicit random seed enabling a fully reproducible simulation and 
  Numpy's default random generator.
 
  @param n the number of vertices in the graph
  @param expectedDegree the target expected degree per vertex
  @return an adjacency list representing the undirected graph
  @throws ValueError if parameters are invalid
  
  Note: I am aware that the the paper uses the matrix format. However, it is VERY inefficient. Therefore, I have changed 
  to a linked list format.


"""
def generate_adjacency_matrix(n, expected_degree, rng):
    p = expected_degree / n
    adj_list = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i+1 , n):
            if rng.random() < p:
                adj_list[i].append(j)
                adj_list[j].append(i)

    return adj_list