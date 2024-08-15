import json
import random

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from walk_dataset import WalkDataset


class ModelTrainer:
    def __init__(self, output_dir, classifier, tune, train_dataset, valid_dataset, test_dataset, device):
        self.output_dir = output_dir
        self.tune = tune
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.best_params = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        predictions = np.argmax(logits, axis=-1)
        eval_accuracy = accuracy_score(labels, predictions)
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        roc_auc = roc_auc_score(labels, probabilities)

        # Generate the classification report
        class_report = classification_report(labels, predictions, output_dict=True)

        return {
            "eval_accuracy": eval_accuracy,
            "roc_auc": roc_auc,
            "classification_report": class_report
        }

    def get_optimizer(self, params):
        lr = params['learning_rate'] if params else self.best_params['learning_rate']
        return SGD(self.classifier.model.parameters(), lr=lr)

    def get_trainer(self, args, train_dataset, eval_dataset, params=None):
        return Trainer(
            model=self.classifier.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.classifier.tokenizer,
            data_collator=WalkDataset.collate_fn,
            compute_metrics=self.compute_metrics,
            optimizers=(self.get_optimizer(params), None)  # Pass SGD as the optimizer
        )

    def create_data_loaders(self, batch_size):
        # Prepare data loaders from datasets
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=WalkDataset.collate_fn)
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=WalkDataset.collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=WalkDataset.collate_fn)
        return train_loader, valid_loader, test_loader

    def tune_hyperparameters(self, n_iter=10):
        param_distributions = {
            'learning_rate': np.logspace(-5, -4, num=100),  # 1e-5 to 1e-4
            'num_train_epochs': [3, 5, 7, 9],
            'per_device_train_batch_size': [16, 32],
            'warmup_ratio': [0.0, 0.1, 0.2]
        }
        best_params = None
        best_score = float('-inf')

        for _ in range(n_iter):
            # Randomly sample hyperparameters
            params = {
                'learning_rate': random.choice(param_distributions['learning_rate']),
                'num_train_epochs': random.choice(param_distributions['num_train_epochs']),
                'per_device_train_batch_size': random.choice(param_distributions['per_device_train_batch_size']),
                'warmup_ratio': random.choice(param_distributions['warmup_ratio']),
            }
            batch_size = params['per_device_train_batch_size']
            total_steps = len(self.train_dataset) // batch_size * params['num_train_epochs']
            warmup_steps = int(total_steps * params['warmup_ratio'])

            print(f"Trying parameters: {params} with warmup_steps={warmup_steps}")

            train_loader, valid_loader, _ = self.create_data_loaders(batch_size)

            training_args = TrainingArguments(
                output_dir=f'{self.output_dir}/fine_tuning',
                evaluation_strategy='epoch',  # Evaluate at the end of each epoch
                save_strategy='epoch',  # Save a checkpoint at the end of each epoch
                save_total_limit=10,  # Only keep the last 10 checkpoints
                learning_rate=params['learning_rate'],
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=params['num_train_epochs'],
                weight_decay=0.01,
                warmup_steps=warmup_steps,
                logging_strategy="epoch",  # ELog training data stats for loss
                logging_dir=f'{self.output_dir}/fine_tuning/logs',
            )

            trainer = self.get_trainer(training_args, train_loader.dataset, valid_loader.dataset, params)
            trainer.train()
            eval_results = trainer.evaluate()

            print(f"Evaluation results: {eval_results}")

            if eval_results['eval_accuracy'] > best_score:
                best_score = eval_results['eval_accuracy']
                best_params = params

        self.best_params = best_params

        # Save the best parameters using json.dump
        with open(f'{self.output_dir}/best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        print(f"Best parameters found: {best_params} with accuracy {best_score}")
        return best_params

    def train(self):
        if self.tune:
            print("Tuning hyperparameters...")
            self.tune_hyperparameters()
        else:
            # If not tuning, set default best_params
            self.best_params = {
                'learning_rate': 2e-5,
                'num_train_epochs': 5,
                'per_device_train_batch_size': 16,
                'warmup_ratio': 0.1
            }

        batch_size = self.best_params['per_device_train_batch_size']
        total_steps = len(self.train_dataset) // batch_size * self.best_params['num_train_epochs']
        warmup_steps = int(self.best_params.get('warmup_ratio', 0.1) * total_steps)

        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/training",
            evaluation_strategy='epoch',  # Evaluate at the end of each epoch
            save_strategy='epoch',  # Save a checkpoint at the end of each epoch
            learning_rate=self.best_params['learning_rate'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=self.best_params['num_train_epochs'],
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            logging_dir=f'{self.output_dir}/training/logs',
            logging_strategy="epoch",
        )

        # Initialize early stopping callback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

        train_loader, valid_loader, test_loader = self.create_data_loaders(batch_size)

        trainer = self.get_trainer(training_args, train_loader.dataset, valid_loader.dataset)

        # Add the early stopping callback
        trainer.add_callback(early_stopping)

        # Train the model
        print("Training the model...")
        trainer.train()

        # Save final model
        trainer.save_model(f'{self.output_dir}/final_model')
        self.classifier.tokenizer.save_pretrained(f'{self.output_dir}/final_model')

        # Perform evaluation on test data
        print("Evaluating the model on the test dataset...")
        test_results = self.evaluate(test_loader)

        # Save final evaluation results using json.dump
        with open(f"{self.output_dir}/final_evaluation_results.json", "w") as f:
            json.dump(test_results, f, indent=4)

        print(f"Evaluation results on test dataset: {test_results}")
        return test_results

    def evaluate(self, data_loader):
        self.classifier.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.classifier.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self.compute_metrics((all_logits, all_labels))
        return metrics
