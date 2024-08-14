import random
from random_walk_classifier.adapted_ergrw_random_walk import AdaptedERGRWRandomWalk
from random_walk_classifier.graph import create_sample_graph


def main():
    random.seed(40)  # For reproducibility

    G = create_sample_graph()
    alpha = 0.5  # Probability of choosing rule1 or rule2

    walker = AdaptedERGRWRandomWalk(G, alpha)
    num_walks = 5
    walk_length = 5

    walks = walker.generate_walks(num_walks, walk_length)

    print("Generated Random Walks:")
    for i, walk in enumerate(walks):
        print(f"Walk {i + 1}: {' '.join(map(str, walk))}")


if __name__ == "__main__":
    main()
