import json
import os
import random
import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import logging

from transformers import AdamW

from walk_dataset import WalkDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, output_dir, classifier, tune, train_dataset, valid_dataset, test_dataset, optimizer_choice,
                 n_iterations, early_drop, device, threshold=0.5):
        self.output_dir = output_dir
        self.tune = tune
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.best_params = None
        self.best_eval_accuracy = 0.0
        self.n_iterations = n_iterations
        self.early_drop = early_drop
        self.early_stopping_patience = 3  # Define early stopping patience
        self.optimizer_choice = optimizer_choice
        self.threshold = threshold  # Decision threshold

    def compute_metrics(self, logits, labels):
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        eval_accuracy = accuracy_score(labels, predictions)
        roc_auc = roc_auc_score(labels, probabilities)
        class_report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(labels, predictions)

        return {
            "eval_accuracy": eval_accuracy,
            "roc_auc": roc_auc,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist()
        }

    def create_data_loaders(self, batch_size):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=WalkDataset.collate_fn)
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=WalkDataset.collate_fn)
        return train_loader, valid_loader

    def tune_hyperparameters(self):
        param_distributions = {
            'learning_rate': [5e-5, 4e-5, 3e-5, 2e-5],
            'num_train_epochs': [2, 3, 4] if self.optimizer_choice == "bertadam" else [3, 5, 7, 9],
            'per_device_train_batch_size': [16, 32],
            'warmup_ratio': [0.0, 0.1, 0.2],
        }

        if self.optimizer_choice == "sgd":
            param_distributions['momentum'] = np.linspace(0.8, 0.99, num=20)

        best_params = None
        best_score = float('-inf')

        completed_trials = set()
        completed_trials_file = os.path.join(self.output_dir, 'completed_trials.json')
        if os.path.exists(completed_trials_file):
            with open(completed_trials_file, 'r') as f:
                completed_trials = {tuple(tuple(pair) for pair in trial) for trial in json.load(f)}

        for _ in range(self.n_iterations):
            params = {
                'learning_rate': random.choice(param_distributions['learning_rate']),
                'num_train_epochs': random.choice(param_distributions['num_train_epochs']),
                'per_device_train_batch_size': random.choice(param_distributions['per_device_train_batch_size']),
                'warmup_ratio': random.choice(param_distributions['warmup_ratio']),
            }

            if self.optimizer_choice == "sgd":
                params['momentum'] = random.choice(param_distributions['momentum'])

            params_tuple = tuple(params.items())
            if params_tuple in completed_trials:
                logger.info(f"Skipping completed parameters: {params}")
                continue

            batch_size = params['per_device_train_batch_size']
            total_steps = len(self.train_dataset) // batch_size * params['num_train_epochs']
            warmup_steps = int(total_steps * params['warmup_ratio'])

            logger.info(f"Trying parameters: {params} with warmup_steps={warmup_steps}")

            if self.optimizer_choice == "sgd":
                self.train(epochs=params['num_train_epochs'],
                           batch_size=batch_size,
                           learning_rate=params['learning_rate'],
                           momentum=params['momentum'],
                           warmup_steps=warmup_steps)
            else:
                self.train(epochs=params['num_train_epochs'],
                           batch_size=batch_size,
                           learning_rate=params['learning_rate'],
                           warmup_steps=warmup_steps)

            eval_results = self.best_eval_accuracy
            logger.info(f"Evaluation results: {eval_results}")

            if eval_results > best_score:
                best_score = eval_results
                best_params = params

            completed_trials.add(params_tuple)
            with open(completed_trials_file, 'w') as f:
                json.dump([list(k) for k in completed_trials], f)  # Convert tuple to list for JSON serialization

        self.best_params = best_params
        with open(f'{self.output_dir}/best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        logger.info(f"Best parameters found: {best_params} with accuracy {best_score}")
        return best_params

    def train_epoch(self, model, train_loader, optimizer, loss_fn, scheduler=None):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate_epoch(self, model, valid_loader, loss_fn):
        model.eval()
        total_loss = 0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(valid_loader)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = self.compute_metrics(all_logits, all_labels)
        return avg_loss, metrics

    def train(self, epochs=7, batch_size=16, learning_rate=0.0001, momentum=0.91, warmup_steps=0, weight_decay=0.01):
        output_dir_name = (f"{self.output_dir}/training_lr{learning_rate}_"
                           f"epochs{epochs}_"
                           f"batch{batch_size}_"
                           f"momentum{momentum}_" if self.optimizer_choice == "sgd" else ""
                           f"warmup{warmup_steps}")

        os.makedirs(output_dir_name, exist_ok=True)
        logging_dir = f'{output_dir_name}/logs'
        os.makedirs(logging_dir, exist_ok=True)

        train_loader, valid_loader = self.create_data_loaders(batch_size)

        loss_fn = CrossEntropyLoss()

        # Choose the optimizer based on the optimizer_choice argument
        if self.optimizer_choice == "bertadam":
            optimizer = AdamW(self.classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif self.optimizer_choice == "sgd":
            optimizer = SGD(self.classifier.model.parameters(), lr=learning_rate, momentum=momentum,
                            weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer choice: {self.optimizer_choice}. Choose 'BertAdam' or 'SGD'.")

        scheduler = LambdaLR(optimizer,
                             lr_lambda=lambda step: min((step + 1) / warmup_steps, 1)) if warmup_steps > 0 else None

        no_improvement_epochs = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(self.classifier.model, train_loader, optimizer, loss_fn, scheduler)
            valid_loss, metrics = self.validate_epoch(self.classifier.model, valid_loader, loss_fn)

            logger.info(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
            logger.info(f"Validation Metrics: {metrics}")

            eval_accuracy = metrics['eval_accuracy']
            if eval_accuracy > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_accuracy
                # no_improvement_epochs = 0
                self.save_checkpoint(epoch, self.classifier.model, optimizer, scheduler, eval_accuracy, output_dir_name)
            elif self.early_drop:
                no_improvement_epochs += 1
                if no_improvement_epochs >= self.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break

    def load_checkpoint(self, checkpoint_path, warmup_steps, total_steps):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classifier.model.load_state_dict(checkpoint['model_state_dict'])

        # Re-create the optimizer with SGD
        optimizer = SGD(self.classifier.model.parameters(),
                        lr=checkpoint['optimizer_state_dict']['param_groups'][0]['lr'],
                        momentum=checkpoint['optimizer_state_dict']['param_groups'][0]['momentum'],
                        weight_decay=checkpoint['optimizer_state_dict']['param_groups'][0]['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        if checkpoint['scheduler_state_dict']:
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler = None

        logger.info(f"Model loaded from {checkpoint_path}")
        return optimizer, scheduler

    @staticmethod
    def save_checkpoint(epoch, model, optimizer, scheduler, eval_accuracy, output_dir_name):
        checkpoint_dir = os.path.join(output_dir_name, f"checkpoint_epoch_{epoch + 1}_acc_{eval_accuracy:.4f}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'eval_accuracy': eval_accuracy
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pth'))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def predict_link(self, entity1, entity2, relation):
        """
        Predicts the likelihood that a link (relation) exists between two entities.

        Args:
            entity1 (str): The first entity.
            entity2 (str): The second entity.
            relation (str): The relation between entity1 and entity2.

        Returns:
            float: The probability that the link exists.
        """
        # Prepare the input text for BERT
        input_text = f"{entity1} {relation} {entity2}"

        # Tokenize the input text using the BERT tokenizer
        inputs = self.classifier.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)

        # Move inputs to the appropriate device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Set the model to evaluation mode
        self.classifier.model.eval()

        # Perform the forward pass to get logits
        with torch.no_grad():
            outputs = self.classifier.model(**inputs)
            logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        # Assuming binary classification (link exists vs. link doesn't exist)
        link_prob = probabilities[0][1]  # Probability of the link existing

        return link_prob

    def evaluate(self, batch_size=16):
        # Create a DataLoader for the test dataset
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=WalkDataset.collate_fn)

        # Set the model to evaluation mode
        self.classifier.model.eval()

        # Initialize variables to accumulate loss and predictions
        total_loss = 0
        all_logits = []
        all_labels = []

        # Iterate over the test dataset
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get the model's predictions
                outputs = self.classifier.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute the loss
                loss = CrossEntropyLoss()(logits, labels)  # Corrected to apply the loss function correctly
                total_loss += loss.item()

                # Collect the logits and labels for metric computation
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Compute the average loss over the test dataset
        avg_loss = total_loss / len(test_loader)

        # Concatenate all predictions and labels
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute evaluation metrics
        test_metrics = self.compute_metrics(all_logits, all_labels)
        test_metrics["test_loss"] = avg_loss  # Add the average loss to the metrics

        # Print the test metrics
        logger.info(f"Test Metrics: {test_metrics}")

        # Save the final evaluation results to a JSON file
        final_results_path = os.path.join(self.output_dir, "final_evaluation_results.json")
        with open(final_results_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Evaluation results saved to {final_results_path}")

        return test_metrics

    def run(self):
        if self.tune:
            best_params = self.tune_hyperparameters()
            logger.info(f"Best hyperparameters: {best_params}")

            # Recover the best model
            output_dir_name = (f"{self.output_dir}/training_lr{best_params['learning_rate']}_"
                               f"epochs{best_params['num_train_epochs']}_"
                               f"batch{best_params['per_device_train_batch_size']}_"
                               f"warmup{best_params['warmup_ratio']}")
            checkpoint_dir = os.path.join(output_dir_name, f"checkpoint_epoch_{self.best_eval_accuracy:.4f}")
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

            # Calculate warmup steps and total steps
            batch_size = best_params['per_device_train_batch_size']
            total_steps = len(self.train_dataset) // batch_size * best_params['num_train_epochs']
            warmup_steps = int(total_steps * best_params['warmup_ratio'])

            optimizer, scheduler = self.load_checkpoint(checkpoint_path, warmup_steps, total_steps)

            # Evaluate the model with the best hyperparameters
            self.evaluate(batch_size=batch_size)
        else:
            self.train()
            self.evaluate()
