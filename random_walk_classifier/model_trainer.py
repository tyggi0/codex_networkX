import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

from walk_dataset import WalkDataset


class ModelTrainer:
    def __init__(self, classifier, train_dataset, valid_dataset, test_dataset, device, batch_size=32, tune=False):
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size = batch_size
        self.tune = tune
        self.best_params = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        accuracy = accuracy_score(torch.tensor(labels), predictions)  # TODO F1, confusion matrix (if necessry)
        return {"eval_accuracy": accuracy}

    def get_trainer(self, args, train_dataset, eval_dataset):
        return Trainer(
            model=self.classifier.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.classifier.tokenizer,
            data_collator=WalkDataset.collate_fn,
            compute_metrics=self.compute_metrics
        )

    def create_data_loaders(self, batch_size):
        # Prepare data loaders from datasets
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=WalkDataset.collate_fn)
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=WalkDataset.collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=WalkDataset.collate_fn)
        return train_loader, valid_loader, test_loader

    def tune_hyperparameters(self):
        param_grid = {
            'learning_rate': [1e-5, 3e-5, 5e-5],
            'num_train_epochs': [2, 3, 4],
            'per_device_train_batch_size': [16, 32],
        }
        best_params = None
        best_score = float('-inf')

        for params in ParameterGrid(param_grid):
            print(f"Trying parameters: {params}")
            train_loader, valid_loader, _ = self.create_data_loaders(params['per_device_train_batch_size'])

            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                logging_strategy="epoch",  # ELog training data stats for loss
                learning_rate=params['learning_rate'],
                per_device_train_batch_size=params['per_device_train_batch_size'],
                per_device_eval_batch_size=32,
                num_train_epochs=params['num_train_epochs'],
                weight_decay=0.01,
                logging_dir='./logs',
            )

            trainer = self.get_trainer(training_args, train_loader.dataset, valid_loader.dataset)
            trainer.train()
            eval_results = trainer.evaluate()

            print(f"Evaluation results: {eval_results}")

            if eval_results['eval_accuracy'] > best_score:
                best_score = eval_results['eval_accuracy']
                best_params = params

        print(f"Best parameters found: {best_params} with accuracy {best_score}")
        return best_params

    def train(self):
        if self.tune:
            self.tune_hyperparameters()
            batch_size = self.best_params['per_device_train_batch_size']
        else:
            batch_size = self.batch_size

        train_loader, valid_loader, test_loader = self.create_data_loaders(batch_size)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,  # Got 0.5 accurary with 1e-4
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=5,  # was 3
            weight_decay=0.01,
            warmup_steps=500,  # Adding warmup steps
        )

        if self.tune:
            print("Tuning hyperparameters...")
            best_params = self.tune_hyperparameters()
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=best_params['learning_rate'],
                per_device_train_batch_size=best_params['per_device_train_batch_size'],
                per_device_eval_batch_size=32,
                num_train_epochs=best_params['num_train_epochs'],
                weight_decay=0.01,
            )

        trainer = self.get_trainer(training_args, train_loader.dataset, valid_loader.dataset)

        # Train the model
        print("Training the model...")
        trainer.train()

        # Evaluate the model on test data
        print("Evaluating the model on test dataset...")
        test_results = trainer.evaluate(eval_dataset=test_loader.dataset)
        print(f"Test dataset evaluation results: {test_results}")

        return test_results
