"""Model training pipeline."""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Train music generation models."""

    def __init__(self, model, model_type: str = 'tensorflow'):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            model_type: 'tensorflow' or 'pytorch'
        """
        self.model = model
        self.model_type = model_type
        self.history = None

    def prepare_data(self, sequences: np.ndarray, 
                    test_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            sequences: Input sequences of shape (num_sequences, sequence_length)
            test_split: Ratio for test split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Split sequences into input and target
        X = sequences[:, :-1]  # All but last note
        y = sequences[:, -1]   # Last note

        # Train-test split
        split_idx = int(len(X) * (1 - test_split))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Data prepared: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def train_tensorflow(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None,
                        epochs: int = 50, batch_size: int = 32) -> dict:
        """
        Train TensorFlow/Keras model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")

        model = self.model.get_model()

        val_data = (X_val, y_val) if X_val is not None else None

        history = model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        self.history = history.history
        logger.info("Training complete")
        return self.history

    def train_pytorch(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None,
                     epochs: int = 50, batch_size: int = 32,
                     learning_rate: float = 0.001) -> dict:
        """
        Train PyTorch model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.LongTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history['loss'].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.LongTensor(X_val).to(device)
                    y_val_tensor = torch.LongTensor(y_val).to(device)
                    
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    history['val_loss'].append(val_loss.item())

            if (epoch + 1) % 10 == 0:
                val_loss_str = f", val_loss: {history['val_loss'][-1]:.4f}" if X_val is not None else ""
                logger.info(f"Epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}{val_loss_str}")

        self.history = history
        logger.info("Training complete")
        return history

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 50, batch_size: int = 32,
             learning_rate: float = 0.001) -> dict:
        """
        Train model (framework-agnostic).
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if self.model_type == 'tensorflow':
            return self.train_tensorflow(X_train, y_train, X_val, y_val, epochs, batch_size)
        elif self.model_type == 'pytorch':
            return self.train_pytorch(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def save_model(self, path: str):
        """Save trained model."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model."""
        self.model.load(path)
        logger.info(f"Model loaded from {path}")
