"""Configuration management for music generation project."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = "./data"
    sequence_length: int = 50
    test_split: float = 0.2
    val_split: float = 0.15
    max_files: Optional[int] = None  # Limit for testing


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = "lstm"  # 'lstm' or 'transformer'
    framework: str = "tensorflow"  # 'tensorflow' or 'pytorch'
    
    # Embedding
    embedding_dim: int = 128
    
    # LSTM specific
    lstm_units: int = 256
    num_lstm_layers: int = 2
    
    # Transformer specific
    num_heads: int = 8
    ff_dim: int = 512
    num_transformer_layers: int = 4
    
    # Common
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    save_checkpoint: bool = True
    checkpoint_dir: str = "./checkpoints"


@dataclass
class GenerationConfig:
    """Configuration for music generation."""
    temperature: float = 1.0  # 0.0 (deterministic) to 2.0+ (random)
    length: int = 100  # Notes to generate
    seed_length: int = 50  # Seed sequence length


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    flask_port: int = 5000
    flask_debug: bool = True
    
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    
    output_dir: str = "./outputs"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    deployment: DeploymentConfig
    
    # Project paths
    project_dir: Path = Path(".")
    models_dir: Path = Path("./models")
    
    def save(self, path: str = "config.json"):
        """Save configuration to JSON file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'generation': self.generation.__dict__,
            'deployment': self.deployment.__dict__,
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str = "config.json"):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            generation=GenerationConfig(**config_dict['generation']),
            deployment=DeploymentConfig(**config_dict['deployment']),
        )


# Default configuration
DEFAULT_CONFIG = Config(
    data=DataConfig(),
    model=ModelConfig(),
    training=TrainingConfig(),
    generation=GenerationConfig(),
    deployment=DeploymentConfig(),
)
