"""Generate sample lecture PDFs with original content using PyMuPDF."""

from pathlib import Path
import fitz


def create_data_structures_pdf(output_path: Path) -> None:
    """Create a sample PDF about data structures."""
    doc = fitz.open()
    
    page1 = doc.new_page()
    text1 = """Data Structures and Algorithms

1. Introduction to Binary Trees

A binary tree is a hierarchical data structure in which each node has at most two children,
referred to as the left child and the right child. Binary trees are fundamental structures
used in many algorithms and data organization schemes.

Properties of binary trees include:
- The maximum number of nodes at level k is 2^k
- A tree with n nodes has exactly n-1 edges
- The height of a complete binary tree with n nodes is log2(n)

Binary trees support efficient operations for searching, insertion, and deletion when properly
balanced. Common applications include expression parsing, decision trees, and hierarchical
data representation."""
    
    page1.insert_text((50, 50), text1, fontsize=11, fontname="helv")
    
    page2 = doc.new_page()
    text2 = """2. Hash Tables and Collision Resolution

Hash tables provide average-case O(1) time complexity for insertion, deletion, and lookup
operations. A hash function maps keys to array indices, enabling direct access to stored
values.

Common collision resolution strategies:
- Chaining: Store colliding elements in linked lists
- Open addressing: Find alternative slots using probing sequences
- Double hashing: Use a secondary hash function for probing

The load factor (number of elements divided by table size) significantly impacts performance.
A load factor between 0.5 and 0.75 typically provides good performance while maintaining
reasonable space efficiency.

Hash tables are widely used in database indexing, caching systems, and symbol tables in
compilers. Modern implementations often use techniques like Robin Hood hashing or Cuckoo
hashing to further improve performance and reduce worst-case behavior."""
    
    page2.insert_text((50, 50), text2, fontsize=11, fontname="helv")
    
    doc.save(str(output_path))
    doc.close()


def create_algorithms_pdf(output_path: Path) -> None:
    """Create a sample PDF about algorithms."""
    doc = fitz.open()
    
    page1 = doc.new_page()
    text1 = """Algorithm Analysis and Design

1. Dynamic Programming Fundamentals

Dynamic programming is an optimization technique that solves complex problems by breaking
them down into simpler subproblems. It is applicable when a problem exhibits optimal
substructure and overlapping subproblems.

Key principles:
- Identify recursive structure of the problem
- Store solutions to subproblems to avoid redundant computation
- Build solutions bottom-up or use memoization top-down

Classic examples include the Fibonacci sequence, longest common subsequence, and the
knapsack problem. Dynamic programming can reduce exponential time complexity to polynomial
time in many cases.

The Bellman equation provides a mathematical foundation for dynamic programming, expressing
the value of a decision problem at a certain point in time in terms of the payoff from
initial choices and the value of the remaining problem."""
    
    page1.insert_text((50, 50), text1, fontsize=11, fontname="helv")
    
    page2 = doc.new_page()
    text2 = """2. Graph Traversal Algorithms

Graph traversal explores all vertices and edges of a graph systematically. The two
fundamental approaches are breadth-first search (BFS) and depth-first search (DFS).

BFS explores nodes level by level using a queue data structure. It finds shortest paths in
unweighted graphs and has applications in network routing and social network analysis.

DFS explores as far as possible along each branch using a stack (or recursion). It is
useful for topological sorting, cycle detection, and solving maze problems. The time
complexity for both algorithms is O(V + E) where V is vertices and E is edges.

Advanced graph algorithms build upon these traversal methods. Dijkstra algorithm extends
BFS to handle weighted graphs, while strongly connected components can be found using
modified DFS. Understanding these fundamental traversal patterns is essential for mastering
more complex graph algorithms."""
    
    page2.insert_text((50, 50), text2, fontsize=11, fontname="helv")
    
    doc.save(str(output_path))
    doc.close()


def main() -> None:
    """Generate all sample PDFs."""
    output_dir = Path("data/sample_lectures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating data_structures.pdf...")
    create_data_structures_pdf(output_dir / "data_structures.pdf")
    
    print("Generating algorithms.pdf...")
    create_algorithms_pdf(output_dir / "algorithms.pdf")
    
    print("Done! Sample PDFs created in data/sample_lectures/")


if __name__ == "__main__":
    main()
