"""Music generation with temperature sampling."""

import logging
from typing import List, Dict, Optional
import numpy as np

try:
    from music21 import stream, instrument, note as m21_note, tempo, meter
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicGenerator:
    """Generate music from trained models."""

    def __init__(self, model, vocab: Dict, model_type: str = 'tensorflow'):
        """
        Initialize generator.
        
        Args:
            model: Trained model
            vocab: Vocabulary mapping (from midi_utils.build_vocabulary)
            model_type: 'tensorflow' or 'pytorch'
        """
        self.model = model
        self.vocab = vocab
        self.model_type = model_type
        self.note_to_index = vocab['note_to_index']
        self.index_to_note = vocab['index_to_note']

    def generate_sequence(self, seed_sequence: List[int], 
                         length: int = 100,
                         temperature: float = 1.0) -> List[int]:
        """
        Generate music sequence with temperature sampling.
        
        Args:
            seed_sequence: Starting sequence of note indices
            length: Length of sequence to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0+ = more random)
            
        Returns:
            Generated sequence of note indices
        """
        generated = seed_sequence.copy()

        for _ in range(length):
            # Get last sequence_length notes
            input_seq = np.array([generated[-len(seed_sequence):]])

            # Predict next note
            if self.model_type == 'tensorflow':
                predictions = self.model.predict(input_seq, verbose=0)[0]
            elif self.model_type == 'pytorch':
                import torch
                input_tensor = torch.LongTensor(input_seq)
                with torch.no_grad():
                    logits = self.model(input_tensor)[0]
                predictions = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Temperature sampling
            next_note = self._sample_with_temperature(predictions, temperature)
            generated.append(next_note)

        return generated

    @staticmethod
    def _sample_with_temperature(probabilities: np.ndarray, 
                                 temperature: float = 1.0) -> int:
        """
        Sample from probability distribution with temperature.
        
        Args:
            probabilities: Probability distribution over vocab
            temperature: Temperature for sampling
                        - 0.0: Always pick highest probability
                        - 1.0: Standard sampling
                        - > 1.0: More random/creative
            
        Returns:
            Sampled index
        """
        if temperature <= 0:
            return np.argmax(probabilities)

        # Apply temperature
        logits = np.log(probabilities + 1e-10) / temperature
        logits = logits - np.max(logits)  # Numerical stability
        
        # Convert to probabilities
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Sample
        return np.random.choice(len(probs), p=probs)

    def indices_to_notes(self, indices: List[int]) -> List[str]:
        """
        Convert note indices to note names.
        
        Args:
            indices: List of note indices
            
        Returns:
            List of note names
        """
        notes = []
        for idx in indices:
            if idx in self.index_to_note:
                note = self.index_to_note[idx]
                # If it's an integer note (MIDI), convert to note name
                if isinstance(note, int):
                    from src.data import MidiUtils
                    note = MidiUtils.index_to_note(note)
                notes.append(note)
        
        return notes

    def generate_midi(self, sequence: List[int],
                     output_path: str,
                     bpm: int = 120,
                     instrument_name: str = 'Piano',
                     duration_quarters: float = 1.0) -> str:
        """
        Convert sequence to MIDI file and save.
        
        Args:
            sequence: Sequence of note indices
            output_path: Path to save MIDI file
            bpm: Tempo in beats per minute
            instrument_name: Instrument for MIDI
            duration_quarters: Duration of each note in quarter notes
            
        Returns:
            Path to saved file
        """
        if not MUSIC21_AVAILABLE:
            raise ImportError("music21 is required. Install with: pip install music21")

        # Create score
        s = stream.Score()
        part = stream.Part()
        
        # Set instrument
        try:
            instr = instrument.Piano()
            part.append(instr)
        except:
            logger.warning("Could not set instrument, using default")

        # Set tempo
        part.append(tempo.MetronomeMark(number=bpm))
        
        # Set time signature
        part.append(meter.TimeSignature('4/4'))

        # Add notes
        for idx in sequence:
            if idx in self.index_to_note:
                note_value = self.index_to_note[idx]
                
                # Handle special tokens
                if isinstance(note_value, str) and note_value.startswith('<'):
                    continue
                
                # Handle rests
                if note_value == -1:
                    part.append(m21_note.Rest(quarterLength=duration_quarters))
                else:
                    # Create note
                    pitch = note_value if isinstance(note_value, int) else 60
                    n = m21_note.Note(pitch, quarterLength=duration_quarters)
                    part.append(n)

        s.append(part)
        s.write('midi', fp=output_path)
        logger.info(f"MIDI file saved to {output_path}")
        
        return output_path

    def generate_and_save(self, seed_sequence: List[int],
                         output_path: str,
                         length: int = 100,
                         temperature: float = 1.0,
                         **midi_kwargs) -> str:
        """
        Generate sequence and save as MIDI.
        
        Args:
            seed_sequence: Seed sequence
            output_path: Output MIDI file path
            length: Length to generate
            temperature: Sampling temperature
            **midi_kwargs: Additional arguments for generate_midi
            
        Returns:
            Path to saved MIDI file
        """
        logger.info(f"Generating sequence with temperature={temperature}...")
        sequence = self.generate_sequence(seed_sequence, length, temperature)
        
        logger.info("Converting to MIDI...")
        return self.generate_midi(sequence, output_path, **midi_kwargs)
