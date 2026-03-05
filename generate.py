"""Music generation script."""

import logging
import argparse
from pathlib import Path
import sys
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.midi_utils import MidiUtils
from src.models.lstm_model import LSTMMusic
from src.models.transformer_model import TransformerMusic
from src.generation.generator import MusicGenerator
from src.visualization.visualizer import MusicVisualizer
from configs.config import DEFAULT_CONFIG, Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main generation pipeline."""
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = DEFAULT_CONFIG
    
    logger.info("=" * 60)
    logger.info("Music Generation - Inference Pipeline")
    logger.info("=" * 60)
    
    models_dir = Path(config.models_dir)
    
    # Load vocabulary
    logger.info("\n[Step 1] Loading vocabulary...")
    vocab_path = models_dir / "vocabulary.json"
    if not vocab_path.exists():
        logger.error(f"Vocabulary not found at {vocab_path}")
        logger.info("Please train a model first using train.py")
        return
    
    vocab = MidiUtils.load_vocabulary(str(vocab_path))
    logger.info(f"Vocabulary loaded. Size: {vocab['vocab_size']}")
    
    # Load model
    logger.info("\n[Step 2] Loading model...")
    model_type = args.model_type or config.model.model_type
    framework = args.framework or config.model.framework
    
    model_path = models_dir / f"{model_type}_{framework}.h5"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.info(f"Available models in {models_dir}:")
        for f in models_dir.glob("*.h5"):
            logger.info(f"  - {f.name}")
        return
    
    if model_type.lower() == "lstm":
        model = LSTMMusic(vocab_size=vocab['vocab_size'])
    elif model_type.lower() == "transformer":
        model = TransformerMusic(vocab_size=vocab['vocab_size'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load(str(model_path))
    logger.info(f"Model loaded: {model_path}")
    
    # Create generator
    logger.info("\n[Step 3] Creating generator...")
    generator = MusicGenerator(model.get_model(), vocab, model_type=framework)
    
    # Generate music
    logger.info("\n[Step 4] Generating music...")
    seed_length = args.seed_length or config.generation.seed_length
    generate_length = args.length or config.generation.length
    temperature = args.temperature or config.generation.temperature
    
    # Create seed sequence (random notes)
    seed = list(np.random.randint(1, vocab['vocab_size'], size=seed_length))
    logger.info(f"Seed: {seed[:10]}... (showing first 10)")
    logger.info(f"Temperature: {temperature}")
    
    sequence = generator.generate_sequence(seed, generate_length, temperature)
    logger.info(f"Generated sequence of length {len(sequence)}")
    
    # Save as MIDI
    logger.info("\n[Step 5] Converting to MIDI...")
    output_dir = Path(args.output_dir or config.deployment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = args.output or f"generated_temp{temperature}.mid"
    output_path = output_dir / output_name
    
    generator.generate_midi(sequence, str(output_path), bpm=args.bpm or 120)
    logger.info(f"MIDI file saved: {output_path}")
    
    # Visualize (optional)
    if args.visualize:
        logger.info("\n[Step 6] Creating visualizations...")
        
        # Piano roll
        visualizer = MusicVisualizer()
        viz_path = output_dir / f"{output_name.replace('.mid', '')}_pianoroll.png"
        visualizer.plot_piano_roll(sequence, title=f"Generated Music (Temp: {temperature})",
                                  output_path=str(viz_path))
        logger.info(f"Visualization saved: {viz_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Generation Complete!")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music from trained model")
    parser.add_argument("--model_type", type=str, choices=['lstm', 'transformer'],
                       help="Model type (lstm or transformer)")
    parser.add_argument("--framework", type=str, choices=['tensorflow', 'pytorch'],
                       help="Deep learning framework")
    parser.add_argument("--seed_length", type=int, help="Seed sequence length")
    parser.add_argument("--length", type=int, help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--bpm", type=int, help="Tempo in BPM")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    main(args)
