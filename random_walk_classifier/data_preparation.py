import re
import logging
from walk_dataset import WalkDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_description(description):
    # Remove anything in parentheses that follows "Property:"
    cleaned = re.sub(r'\s*\(Property:[^)]*\)', '', description)
    return cleaned


class DataPreparation:
    def __init__(self, generator, classifier, codex):
        self.generator = generator
        self.classifier = classifier
        self.codex = codex

    def transform_triples(self, triples):
        transformed = []
        for head, relation, tail in triples.values:
            head_label = f"{self.codex.entity_label(head)}: {clean_description(self.codex.entity_description(head))}"
            relation_label = f"{self.codex.relation_label(relation)}: {clean_description(self.codex.relation_description(relation))}"
            tail_label = f"{self.codex.entity_label(tail)}: {clean_description(self.codex.entity_description(tail))}"
            transformed.append([head_label, relation_label, tail_label])
        return transformed

    @staticmethod
    def encode_walks(walks):
        """
        Encode walks into a format suitable for BERT input, including both labels and descriptions.

        Args:
            walks (list of lists): A list of walks, where each walk is a sequence of alternating entities and relations.

        Returns:
            list of str: A list of encoded strings, one for each walk.
        """
        encoded_walks = []

        for walk in walks:
            encoded_walk = []
            for i, element in enumerate(walk):
                if i % 2 == 0:  # Even index: Entity
                    encoded_walk.append(f"[ENTITY{i // 2 + 1}] {element} [/ENTITY{i // 2 + 1}]")
                else:  # Odd index: Relation
                    encoded_walk.append(f"[RELATION{i // 2 + 1}] {element} [/RELATION{i // 2 + 1}]")

            # Join the parts into a single string for BERT input
            encoded_walks.append(' '.join(encoded_walk))

        return encoded_walks

    def encode_data(self, valid_walks, invalid_walks):
        encoded_valid_walks = self.encode_walks(valid_walks)
        encoded_invalid_walks = self.encode_walks(invalid_walks)

        labels_valid = [1] * len(encoded_valid_walks)
        labels_invalid = [0] * len(encoded_invalid_walks)

        walks = encoded_valid_walks + encoded_invalid_walks
        labels = labels_valid + labels_invalid

        logger.info(f"Total encoded valid walks: {len(encoded_valid_walks)}")
        logger.info(f"Total encoded invalid walks: {len(encoded_invalid_walks)}")
        logger.info(f"Total combined walks: {len(walks)}")

        # Log first 10 valid and invalid walks
        for i, walk in enumerate(encoded_valid_walks[:10]):
            logger.info(f"Valid Walk {i + 1}: {walk}")
        for i, walk in enumerate(encoded_invalid_walks[:10]):
            logger.info(f"Invalid Walk {i + 1}: {walk}")

        return walks, labels

    def prepare_train_dataset(self, random_walk_strategy, alpha, num_walks, walk_length):
        # Load the Codex train split
        train_triples = self.codex.split("train")

        # Assume equal proportions for valid and invalid walks for simplicity
        total_size = len(train_triples)
        sample_size = total_size // 2  # Each category gets half of half (half the dataset, equally split)

        # Sampling for valid walks
        valid_triples_sampled = train_triples.sample(n=sample_size)
        train_valid_walks = self.transform_triples(valid_triples_sampled)

        # Corrupt equivalent amount for invalid walks, ensuring stratified sampling
        remaining_triples = train_triples.drop(valid_triples_sampled.index)
        invalid_triples_sampled = remaining_triples.sample(n=sample_size)
        train_invalid_walks = self.generator.generate_invalid_random_walks(
            self.transform_triples(invalid_triples_sampled))

        logger.info("\nEncoded Train Walks:")
        train_walks, train_labels = self.encode_data(train_valid_walks, train_invalid_walks)

        if random_walk_strategy:
            # Generate additional random walks if a strategy is provided
            valid_walks = self.generator.generate_random_walks(random_walk_strategy, alpha, num_walks, walk_length)
            invalid_walks = self.generator.generate_invalid_random_walks(valid_walks)

            logger.info("\nEncoded Train **Random** Walks:")
            random_walks, labels = self.encode_data(valid_walks, invalid_walks)

            # Combine the walks and labels from the Codex split and random walks
            combined_walks = random_walks + train_walks
            combined_labels = labels + train_labels

            dataset = WalkDataset(combined_walks, combined_labels, self.classifier.tokenizer)

        else:
            logger.info("No random walk strategy provided, using only Codex dataset.")
            dataset = WalkDataset(train_walks, train_labels, self.classifier.tokenizer)

        return dataset

    def prepare_eval_dataset(self, split):
        valid_triples = self.codex.split(split)
        invalid_triples = self.codex.negative_split(split)

        valid_walks = self.transform_triples(valid_triples)
        invalid_walks = self.transform_triples(invalid_triples)

        logger.info(f"\nEncoded {split.capitalize()} Walks:")
        walks, labels = self.encode_data(valid_walks, invalid_walks)

        dataset = WalkDataset(walks, labels, self.classifier.tokenizer)
        return dataset

    def prepare_datasets(self, random_walk_strategy, alpha, num_walks, walk_length):
        logger.info("Preparing datasets...")

        # Prepare training dataset from the graph
        train_dataset = self.prepare_train_dataset(random_walk_strategy, alpha, num_walks, walk_length)

        # Prepare validation dataset from Codex splits
        valid_dataset = self.prepare_eval_dataset("valid")

        # Prepare test dataset from Codex splits
        test_dataset = self.prepare_eval_dataset("test")

        return train_dataset, valid_dataset, test_dataset
