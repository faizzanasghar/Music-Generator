"""Visualization utilities for music analysis and generation."""

import logging
from typing import List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicVisualizer:
    """Visualize music as piano rolls and other plots."""

    @staticmethod
    def plot_piano_roll(notes: List[int], durations: Optional[List[float]] = None,
                       title: str = "Piano Roll", figsize: tuple = (12, 6),
                       output_path: Optional[str] = None):
        """
        Plot a piano roll visualization of notes.
        
        Args:
            notes: List of MIDI note numbers
            durations: List of note durations (optional)
            title: Title of plot
            figsize: Figure size
            output_path: Path to save figure (if provided)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")

        fig, ax = plt.subplots(figsize=figsize)

        # Create piano roll representation
        if durations is None:
            durations = [1.0] * len(notes)

        # Plot each note
        for i, (note, duration) in enumerate(zip(notes, durations)):
            if 0 <= note <= 127:  # Valid MIDI note
                ax.barh(note, duration, left=i, height=0.8, color='steelblue', edgecolor='black')

        # Configure plot
        ax.set_xlabel('Time (quarter notes)')
        ax.set_ylabel('MIDI Note Number')
        ax.set_title(title)
        ax.set_xlim(0, len(notes))
        ax.set_ylim(0, 128)

        # Add note labels
        note_labels = ['C-1', 'C#-1', 'D-1', 'D#-1', 'E-1', 'F-1', 'F#-1', 'G-1', 'G#-1', 'A-1', 'A#-1', 'B-1']
        note_names = []
        for octave in range(-1, 11):
            note_names.extend([f"{name}{octave}" for name in note_labels])

        # Set y-tick labels (show every 12 notes for clarity)
        ax.set_yticks(range(0, 128, 12))
        ax.set_yticklabels([note_names[i] if i < len(note_names) else f"Note {i}" for i in range(0, 128, 12)])

        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Piano roll saved to {output_path}")

        plt.show()

    @staticmethod
    def plot_note_distribution(notes: List[int], title: str = "Note Distribution",
                              figsize: tuple = (12, 6),
                              output_path: Optional[str] = None):
        """
        Plot histogram of note frequencies.
        
        Args:
            notes: List of MIDI note numbers
            title: Title of plot
            figsize: Figure size
            output_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required")

        fig, ax = plt.subplots(figsize=figsize)

        # Count note frequencies
        unique_notes, counts = np.unique(notes, return_counts=True)

        # Create bar plot
        ax.bar(unique_notes, counts, color='steelblue', edgecolor='black')

        ax.set_xlabel('MIDI Note Number')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Note distribution plot saved to {output_path}")

        plt.show()

    @staticmethod
    def plot_training_history(history: dict, title: str = "Training History",
                             figsize: tuple = (12, 4),
                             output_path: Optional[str] = None):
        """
        Plot training loss and accuracy curves.
        
        Args:
            history: Training history dictionary
            title: Title of plot
            figsize: Figure size
            output_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Loss curve
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', color='blue')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', color='orange')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curve (if available)
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy', color='green')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training history plot saved to {output_path}")

        plt.show()

    @staticmethod
    def compare_sequences(sequences: dict, title: str = "Sequence Comparison",
                         figsize: tuple = (14, 8),
                         output_path: Optional[str] = None):
        """
        Compare multiple sequences as piano rolls.
        
        Args:
            sequences: Dictionary of {name: notes_list}
            title: Title of plot
            figsize: Figure size
            output_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required")

        n_sequences = len(sequences)
        fig, axes = plt.subplots(n_sequences, 1, figsize=figsize)

        if n_sequences == 1:
            axes = [axes]

        for ax, (name, notes) in zip(axes, sequences.items()):
            # Plot piano roll
            for i, note in enumerate(notes):
                if 0 <= note <= 127:
                    ax.barh(note, 1, left=i, height=0.8, color='steelblue')

            ax.set_ylabel('Note')
            ax.set_title(name)
            ax.set_ylim(0, 128)
            ax.grid(True, alpha=0.3, axis='x')

        fig.suptitle(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_path}")

        plt.show()
