import re
import logging

from walk_dataset import WalkDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_textual_representation(codex, head, relation, tail, description, lowercase):
    def clean_description(text):
        # Remove anything in parentheses that follows "Property:"
        cleaned = re.sub(r'\s*\(Property:[^)]*\)', '', text)
        return cleaned

    if description:
        # Use both labels and descriptions
        head_label = f"{codex.entity_label(head)}: {clean_description(codex.entity_description(head))}"
        relation_label = f"{codex.relation_label(relation)}: {clean_description(codex.relation_description(relation))}"
        tail_label = f"{codex.entity_label(tail)}: {clean_description(codex.entity_description(tail))}"
    else:
        # Use only labels
        head_label = f"{codex.entity_label(head)}"
        relation_label = f"{codex.relation_label(relation)}"
        tail_label = f"{codex.entity_label(tail)}"

    # Convert to lowercase if the option is enabled
    if lowercase:
        head_label = head_label.lower()
        relation_label = relation_label.lower()
        tail_label = tail_label.lower()

    return head_label, relation_label, tail_label


class DataPreparation:
    def __init__(self, generator, classifier, codex, description, encoding_format, lowercase):
        self.generator = generator
        self.classifier = classifier
        self.codex = codex
        self.description = description
        self.encoding_format = encoding_format  # The format in which to encode the walks. Options are "tag" and "bert".
        self.lowercase = lowercase

    def transform_triples(self, triples):
        transformed = []
        for head, relation, tail in triples.values:
            transformed.append(list(create_textual_representation(self.codex, head, relation, tail,
                                                                  self.description, self.lowercase)))
        return transformed

    def encode_walks(self, walks):
        """
        Encode walks into a format suitable for BERT input.

        Args:
            walks (list of lists): A list of walks, where each walk is a sequence of alternating entities and relations.
        Returns:
            list of str: A list of encoded strings, one for each walk.
        """
        encoded_walks = []

        if self.encoding_format == "tag":
            for walk in walks:
                encoded_walk = []
                for i, element in enumerate(walk):
                    if i % 2 == 0:  # Even index: Entity
                        encoded_walk.append(f"[ENTITY{i // 2 + 1}] {element} [/ENTITY{i // 2 + 1}]")
                        encoded_walk.append("[SEP]")
                    else:  # Odd index: Relation
                        encoded_walk.append(f"[RELATION{i // 2 + 1}] {element} [/RELATION{i // 2 + 1}]")
                        encoded_walk.append("[SEP]")

                # Join the parts into a single string for BERT input
                encoded_walks.append(' '.join(encoded_walk))

        elif self.encoding_format == "bert":
            for walk in walks:
                encoded_walk = []

                for i, element in enumerate(walk):
                    encoded_walk.append(element)
                    if i < len(walk) - 1:  # Check if it's not the last element
                        encoded_walk.append("[SEP]")  # Add [SEP] only between elements

                # Join the parts into a single string for BERT input
                encoded_walks.append(' '.join(encoded_walk))

        else:
            raise ValueError("Unknown format specified. Supported formats are 'tag' and 'bert'.")

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

    def prepare_train_dataset(self, random_walk_strategy, alpha, num_walks, walk_length, size, mode):
        """
        Prepare the training dataset based on the specified mode.

        Args:
            random_walk_strategy: The strategy to use for random walks.
            alpha: The alpha parameter for random walk generation.
            num_walks: Number of walks to generate.
            walk_length: Length of each walk.
            size: Dataset size ('full' or 'half').
            mode: Dataset preparation mode ('codex_only', 'random_walks_only', or 'combined').

        Returns:
            WalkDataset: The prepared dataset.
        """
        if mode == "codex_only" or mode == "combined":
            # Load the Codex train split
            train_triples = self.codex.split("train")

            # Assume equal proportions for valid and invalid walks for simplicity
            total_size = len(train_triples)
            if size == "full":
                sample_size = total_size // 2
            elif size == "half":
                sample_size = total_size // 4  # Each category gets half of half (half the dataset, equally split)
            else:
                raise ValueError("Unknown size specified. Supported sizes are 'full' and 'half'.")

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
        else:
            train_walks, train_labels = [], []

        if mode == "random_walks_only" or mode == "combined":
            if random_walk_strategy:
                # Generate additional random walks
                valid_walks = self.generator.generate_random_walks(random_walk_strategy, alpha, num_walks, walk_length, mode)
                invalid_walks = self.generator.generate_invalid_random_walks(valid_walks)

                logger.info("\nEncoded Train **Random** Walks:")
                random_walks, labels = self.encode_data(valid_walks, invalid_walks)

                # Combine the walks and labels from the Codex split and random walks if in combined mode
                if mode == "combined":
                    combined_walks = random_walks + train_walks
                    combined_labels = labels + train_labels
                else:
                    combined_walks = random_walks
                    combined_labels = labels
            else:
                raise ValueError("Random walk strategy must be provided for 'random_walks_only' or 'combined' mode.")
        else:
            combined_walks, combined_labels = train_walks, train_labels

        dataset = WalkDataset(combined_walks, combined_labels, self.classifier.tokenizer)
        logger.info(f"Creating WalkDataset with {len(combined_walks)} walks and {len(combined_labels)} labels.")
        # Log the first item from the dataset
        logger.info(f"First dataset item: {dataset[0]}")

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

    def prepare_datasets(self, random_walk_strategy, alpha, num_walks, walk_length, size, dataset_mode):
        logger.info("Preparing datasets...")

        # Prepare training dataset from the graph
        train_dataset = self.prepare_train_dataset(
            random_walk_strategy, alpha, num_walks, walk_length, size, dataset_mode)

        # Prepare validation dataset from Codex splits
        valid_dataset = self.prepare_eval_dataset("valid")

        # Prepare test dataset from Codex splits
        test_dataset = self.prepare_eval_dataset("test")

        return train_dataset, valid_dataset, test_dataset
