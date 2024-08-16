from walk_dataset import WalkDataset


class DataPreparation:
    def __init__(self, generator, classifier, codex):
        self.generator = generator
        self.classifier = classifier
        self.codex = codex

    def transform_triples(self, triples):
        transformed = []
        for head, relation, tail in triples.values:
            head_label = self.codex.entity_label(head)
            relation_label = self.codex.relation_label(relation)
            tail_label = self.codex.entity_label(tail)
            transformed.append([head_label, relation_label, tail_label])
        return transformed

    @staticmethod
    def encode_walks(walks):
        return [' '.join(map(str, walk)) for walk in walks]

    def encode_data(self, valid_walks, invalid_walks):
        encoded_valid_walks = self.encode_walks(valid_walks)
        encoded_invalid_walks = self.encode_walks(invalid_walks)

        labels_valid = [1] * len(encoded_valid_walks)
        labels_invalid = [0] * len(encoded_invalid_walks)

        walks = encoded_valid_walks + encoded_invalid_walks
        labels = labels_valid + labels_invalid

        print(f"Total encoded valid walks: {len(encoded_valid_walks)}")
        print(f"Total encoded invalid walks: {len(encoded_invalid_walks)}")
        print(f"Total combined walks: {len(walks)}")

        # Print first 10 valid and invalid walks
        for i, walk in enumerate(encoded_valid_walks[:10]):
            print(f"Valid Walk {i + 1}: {walk}")
        for i, walk in enumerate(encoded_invalid_walks[:10]):
            print(f"Invalid Walk {i + 1}: {walk}")

        return walks, labels

    def prepare_train_dataset(self, random_walk_strategy, alpha, num_walks, walk_length):
        # Load the Codex train split
        train_triples = self.codex.split("train")

        # Assume equal proportions for valid and invalid walks for simplicity
        total_size = len(train_triples)
        sample_size = total_size // 4  # Each category gets half of half (half the dataset, equally split)

        # Sampling for valid walks
        valid_triples_sampled = train_triples.sample(n=sample_size)
        train_valid_walks = self.transform_triples(valid_triples_sampled)

        # Corrupt equivalent amount for invalid walks, ensuring stratified sampling
        remaining_triples = train_triples.drop(valid_triples_sampled.index)
        invalid_triples_sampled = remaining_triples.sample(n=sample_size)
        train_invalid_walks = self.generator.generate_invalid_random_walks(
            self.transform_triples(invalid_triples_sampled))

        print("\nEncoded Train Walks:")
        train_walks, train_labels = self.encode_data(train_valid_walks, train_invalid_walks)

        if random_walk_strategy:
            # Generate additional random walks if a strategy is provided
            valid_walks = self.generator.generate_random_walks(random_walk_strategy, alpha, num_walks, walk_length)
            invalid_walks = self.generator.generate_invalid_random_walks(valid_walks)

            print("\nEncoded Train **Random** Walks:")
            walks, labels = self.encode_data(valid_walks, invalid_walks)

            # Combine the walks and labels from the Codex split and random walks
            combined_walks = walks + train_walks
            combined_labels = labels + train_labels

            dataset = WalkDataset(combined_walks, combined_labels, self.classifier.tokenizer)

        else:
            print("No random walk strategy provided, using only Codex dataset.")
            dataset = WalkDataset(train_walks, train_labels, self.classifier.tokenizer)

        return dataset

    def prepare_eval_dataset(self, split):
        valid_triples = self.codex.split(split)
        invalid_triples = self.codex.negative_split(split)

        valid_walks = self.transform_triples(valid_triples)
        invalid_walks = self.transform_triples(invalid_triples)

        print(f"\nEncoded {split.capitalize()} Walks:")
        walks, labels = self.encode_data(valid_walks, invalid_walks)

        dataset = WalkDataset(walks, labels, self.classifier.tokenizer)
        return dataset

    def prepare_datasets(self, random_walk_strategy, alpha, num_walks, walk_length):
        print("Preparing datasets...")

        # Prepare training dataset from the graph
        train_dataset = self.prepare_train_dataset(random_walk_strategy, alpha, num_walks, walk_length)

        # Prepare validation dataset from Codex splits
        valid_dataset = self.prepare_eval_dataset("valid")

        # Prepare test dataset from Codex splits
        test_dataset = self.prepare_eval_dataset("test")

        return train_dataset, valid_dataset, test_dataset