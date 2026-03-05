"""Training script for music generation models."""

import logging
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import DatasetLoader
from src.data.midi_utils import MidiUtils
from src.preprocessing.music21_preprocessor import Music21Preprocessor
from src.models.lstm_model import LSTMMusic
from src.models.transformer_model import TransformerMusic
from src.training.trainer import Trainer
from configs.config import DEFAULT_CONFIG, Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main training pipeline."""
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = DEFAULT_CONFIG
    
    logger.info("=" * 60)
    logger.info("Music Generation - Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load MIDI files
    logger.info("\n[Step 1] Loading MIDI files...")
    dataset_loader = DatasetLoader(config.data.data_dir)
    
    # For demo or testing, you can load from a directory
    midi_files = dataset_loader.load_midi_files(args.data_path or config.data.data_dir)
    
    if len(midi_files) == 0:
        logger.error(f"No MIDI files found in {args.data_path or config.data.data_dir}")
        logger.info("To download datasets:")
        logger.info("  - MAESTRO: https://storage.googleapis.com/magenta-datasets/maestro/v3.0.0/maestro-v3.0.0.zip")
        logger.info("  - Lakh: http://codespeaker.com/public/lakh/")
        return
    
    # Limit files for testing
    if config.data.max_files:
        midi_files = midi_files[:config.data.max_files]
    
    logger.info(f"Loaded {len(midi_files)} MIDI files")
    
    # Validate dataset
    valid_files, invalid_files = dataset_loader.validate_dataset(midi_files)
    logger.info(f"Valid: {len(valid_files)}, Invalid: {len(invalid_files)}")
    
    # Step 2: Preprocess MIDI files
    logger.info("\n[Step 2] Preprocessing MIDI files...")
    preprocessor = Music21Preprocessor(config.data.sequence_length)
    sequences, metadata = preprocessor.batch_preprocess(valid_files)
    
    if len(sequences) == 0:
        logger.error("No sequences generated from MIDI files")
        return
    
    logger.info(f"Generated {len(sequences)} sequences")
    
    # Extract all notes to build vocabulary
    all_notes = []
    for seq in sequences:
        all_notes.extend(seq)
    
    # Step 3: Build vocabulary
    logger.info("\n[Step 3] Building vocabulary...")
    vocab = MidiUtils.build_vocabulary(all_notes)
    logger.info(f"Vocabulary size: {vocab['vocab_size']}")
    
    # Save vocabulary
    vocab_path = Path(config.models_dir) / "vocabulary.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    MidiUtils.save_vocabulary(vocab, str(vocab_path))
    
    # Step 4: Prepare training data
    logger.info("\n[Step 4] Preparing training data...")
    trainer_obj = Trainer(
        model=None,  # Will be set later
        model_type=config.model.framework
    )
    X_train, X_test, y_train, y_test = trainer_obj.prepare_data(
        sequences,
        test_split=config.data.test_split
    )
    
    # Step 5: Build model
    logger.info("\n[Step 5] Building model...")
    if config.model.model_type.lower() == "lstm":
        model = LSTMMusic(
            vocab_size=vocab['vocab_size'],
            embedding_dim=config.model.embedding_dim,
            lstm_units=config.model.lstm_units,
            num_lstm_layers=config.model.num_lstm_layers,
            dropout=config.model.dropout
        )
    elif config.model.model_type.lower() == "transformer":
        model = TransformerMusic(
            vocab_size=vocab['vocab_size'],
            embedding_dim=config.model.embedding_dim,
            num_heads=config.model.num_heads,
            ff_dim=config.model.ff_dim,
            num_layers=config.model.num_transformer_layers,
            dropout=config.model.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
    
    model.build(config.data.sequence_length)
    model.summary()
    
    # Step 6: Train model
    logger.info("\n[Step 6] Training model...")
    trainer_obj.model = model
    
    history = trainer_obj.train(
        X_train, y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate
    )
    
    # Step 7: Save model
    logger.info("\n[Step 7] Saving model...")
    model_path = Path(config.models_dir) / f"{config.model.model_type}_{config.model.framework}.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer_obj.save_model(str(model_path))
    
    # Save training history
    import json
    history_path = Path(config.models_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Vocabulary saved: {vocab_path}")
    logger.info(f"History saved: {history_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train music generation model")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to MIDI files directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    main(args)
