"""Music Generation Package - Core module."""

__version__ = "1.0.0"
__author__ = "Music Generation Team"

from src.data import DatasetLoader, MidiUtils
from src.preprocessing import Music21Preprocessor
from src.models import LSTMMusic, TransformerMusic
from src.training import Trainer
from src.generation import MusicGenerator
from src.visualization import MusicVisualizer

__all__ = [
    "DatasetLoader",
    "MidiUtils",
    "Music21Preprocessor",
    "LSTMMusic",
    "TransformerMusic",
    "Trainer",
    "MusicGenerator",
    "MusicVisualizer",
]
