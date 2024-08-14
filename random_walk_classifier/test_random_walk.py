import random
from random_walk_classifier.graph import create_sample_graph
from random_walk_classifier.random_walk_generator import RandomWalkGenerator

random.seed(34)  # For reproducibility

G = create_sample_graph()
alpha = 0.5  # Probability of choosing rule1 or rule2
num_walks = 5
walk_length = 5

generator = RandomWalkGenerator(G)


def main(random_walk_strategy):
    print(f"\nRandom Walk: {random_walk_strategy}")
    walks = generator.generate_random_walks(random_walk_strategy, alpha, num_walks, walk_length)

    print("Generated Random Walks:")
    for i, walk in enumerate(walks):
        print(f"Walk {i + 1}: {' '.join(map(str, walk))}")


if __name__ == "__main__":
    main("traditional")
    main("ergrw")
    main("adapted_ergrw")