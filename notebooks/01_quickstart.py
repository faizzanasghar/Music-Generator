"""
Example: Quick Start - Music Generation

This notebook demonstrates the complete music generation workflow
from data loading to MIDI generation.
"""

# Step 1: Import libraries
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.data import DatasetLoader, MidiUtils
from src.preprocessing import Music21Preprocessor
from src.models import LSTMMusic
from src.training import Trainer
from src.generation import MusicGenerator
from src.visualization import MusicVisualizer
import numpy as np

print("=" * 60)
print("Music Generation - Quick Start Example")
print("=" * 60)

# Step 2: Load MIDI files
print("\n[Step 1] Loading MIDI files...")
dataset_loader = DatasetLoader("./data")

# Point to your MIDI files directory
midi_files = dataset_loader.load_midi_files("./data")  # Change this path!

if len(midi_files) == 0:
    print("❌ No MIDI files found!")
    print("Please download a dataset and place it in ./data/")
    print("Recommended: MAESTRO - https://storage.googleapis.com/magenta-datasets/maestro/v3.0.0/maestro-v3.0.0.zip")
else:
    print(f"✓ Loaded {len(midi_files)} MIDI files")
    
    # Step 3: Preprocess
    print("\n[Step 2] Preprocessing MIDI files...")
    preprocessor = Music21Preprocessor(sequence_length=50)
    sequences, metadata = preprocessor.batch_preprocess(midi_files[:10])  # First 10 files for demo
    
    print(f"✓ Generated {len(sequences)} sequences from {len(metadata)} files")
    
    if len(sequences) > 0:
        # Step 4: Build vocabulary
        print("\n[Step 3] Building vocabulary...")
        all_notes = []
        for seq in sequences:
            all_notes.extend(seq)
        
        vocab = MidiUtils.build_vocabulary(all_notes)
        print(f"✓ Vocabulary size: {vocab['vocab_size']}")
        
        # Step 5: Build and train model
        print("\n[Step 4] Building LSTM model...")
        model = LSTMMusic(
            vocab_size=vocab['vocab_size'],
            embedding_dim=128,
            lstm_units=256,
            num_lstm_layers=2,
            dropout=0.2
        )
        model.build(sequence_length=50)
        print("✓ Model built!")
        
        # Step 6: Prepare data
        print("\n[Step 5] Preparing training data...")
        trainer = Trainer(model, model_type='tensorflow')
        X_train, X_test, y_train, y_test = trainer.prepare_data(sequences, test_split=0.2)
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Step 7: Train (Quick demo - just 2 epochs)
        print("\n[Step 6] Training model (Demo: 2 epochs)...")
        history = trainer.train(
            X_train, y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=2,  # Just for demo - use 50+ for real training
            batch_size=32
        )
        print("✓ Training complete!")
        
        # Step 8: Generate music
        print("\n[Step 7] Generating music...")
        generator = MusicGenerator(model.get_model(), vocab, model_type='tensorflow')
        
        # Create seed (random notes)
        seed = list(np.random.randint(1, vocab['vocab_size'], size=50))
        
        # Generate with different temperatures
        for temp in [0.5, 1.0, 1.5]:
            print(f"\n  Generating with temperature={temp}...")
            sequence = generator.generate_sequence(seed, length=100, temperature=temp)
            
            # Save as MIDI
            output_path = f"./outputs/demo_temp{temp}.mid"
            generator.generate_midi(sequence, output_path)
            print(f"  ✓ Saved to {output_path}")
        
        print("\n" + "=" * 60)
        print("✓ DEMO COMPLETE!")
        print("=" * 60)
        print("\nGenerated files are in ./outputs/")
        print("Play them with any MIDI player!")
        print("\nFor full training, run:")
        print("  python train.py --data_path ./data")
