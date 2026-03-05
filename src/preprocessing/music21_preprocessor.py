"""
Music21-based MIDI preprocessing.
Extracts notes, durations, and offsets for model training.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    from music21 import converter, instrument, note as m21_note, chord as m21_chord
except ImportError:
    raise ImportError("music21 is required. Install with: pip install music21")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Music21Preprocessor:
    """Preprocess MIDI files using music21."""

    def __init__(self, sequence_length: int = 50):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Fixed length for input sequences
        """
        self.sequence_length = sequence_length

    def load_midi(self, midi_path: str):
        """
        Load MIDI file using music21.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            music21 Score object
        """
        try:
            score = converter.parse(midi_path)
            logger.info(f"Loaded MIDI: {midi_path}")
            return score
        except Exception as e:
            logger.error(f"Error loading MIDI {midi_path}: {e}")
            return None

    def extract_notes_and_durations(self, score) -> Tuple[List[int], List[float], List[float]]:
        """
        Extract notes, durations, and offsets from MIDI.
        
        Args:
            score: music21 Score object
            
        Returns:
            Tuple of (note_pitches, durations, offsets)
        """
        notes = []
        durations = []
        offsets = []

        try:
            # Flatten score to get all notes
            for element in score.flatten().notesAndRests:
                if isinstance(element, m21_note.Note):
                    # Single note
                    notes.append(element.pitch.midi)
                    durations.append(element.duration.quarterLength)
                    offsets.append(element.offset)
                elif isinstance(element, m21_chord.Chord):
                    # For chords, take the lowest note (or process all notes)
                    # Option 1: Use lowest note
                    notes.append(min([p.midi for p in element.pitches]))
                    durations.append(element.duration.quarterLength)
                    offsets.append(element.offset)
                elif isinstance(element, m21_note.Rest):
                    # Rest represented as -1
                    notes.append(-1)
                    durations.append(element.duration.quarterLength)
                    offsets.append(element.offset)
        except Exception as e:
            logger.error(f"Error extracting notes: {e}")
            return [], [], []

        return notes, durations, offsets

    def quantize_durations(self, durations: List[float], resolution: int = 4) -> List[int]:
        """
        Quantize note durations to discrete bins.
        
        Args:
            durations: List of note durations (quarter lengths)
            resolution: Number of quantization bins
            
        Returns:
            Quantized durations as integers
        """
        quantized = []
        for d in durations:
            # Map continuous durations to discrete bins
            bin_idx = max(0, min(resolution - 1, round(d * resolution)))
            quantized.append(bin_idx)
        
        return quantized

    def create_sequences(self, notes: List[int], sequence_length: Optional[int] = None) -> np.ndarray:
        """
        Create fixed-length sequences for training.
        
        Args:
            notes: List of note indices
            sequence_length: Length of sequences (default: self.sequence_length)
            
        Returns:
            Array of shape (num_sequences, sequence_length)
        """
        if sequence_length is None:
            sequence_length = self.sequence_length

        sequences = []
        
        for i in range(len(notes) - sequence_length):
            seq = notes[i:i + sequence_length]
            sequences.append(seq)
        
        logger.info(f"Created {len(sequences)} sequences of length {sequence_length}")
        return np.array(sequences)

    def preprocess_file(self, midi_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Complete preprocessing pipeline for a single file.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Tuple of (sequences, metadata)
        """
        # Load MIDI
        score = self.load_midi(midi_path)
        if score is None:
            return None, {}
        
        # Extract notes
        notes, durations, offsets = self.extract_notes_and_durations(score)
        if not notes:
            return None, {}
        
        # Quantize durations
        quantized_durations = self.quantize_durations(durations)
        
        # Create sequences
        sequences = self.create_sequences(notes)
        
        if len(sequences) == 0:
            return None, {}
        
        metadata = {
            'midi_path': str(midi_path),
            'num_notes': len(notes),
            'num_sequences': len(sequences),
            'unique_notes': len(set(notes)),
            'min_note': min(notes),
            'max_note': max(notes)
        }
        
        return sequences, metadata

    def batch_preprocess(self, midi_files: List[str], 
                        skip_errors: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Preprocess multiple MIDI files.
        
        Args:
            midi_files: List of MIDI file paths
            skip_errors: Skip files with errors
            
        Returns:
            Tuple of (all_sequences, all_metadata)
        """
        all_sequences = []
        all_metadata = []
        
        for midi_file in midi_files:
            try:
                sequences, metadata = self.preprocess_file(midi_file)
                if sequences is not None:
                    all_sequences.append(sequences)
                    all_metadata.append(metadata)
            except Exception as e:
                if not skip_errors:
                    raise
                logger.error(f"Error preprocessing {midi_file}: {e}")
        
        if all_sequences:
            combined_sequences = np.vstack(all_sequences)
        else:
            combined_sequences = np.array([])
        
        logger.info(f"Batch processing complete: {len(all_metadata)} files processed")
        return combined_sequences, all_metadata
