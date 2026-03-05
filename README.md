# 🎵 Music Generation AI

> AI-powered music generation using deep learning. Treat music as a language and generate novel compositions using LSTM and Transformer architectures.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow | PyTorch](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-brightgreen.svg)](https://github.com)

## Overview

This project implements a complete music generation pipeline following state-of-the-art methodologies in AI music research:

- **Data Collection**: Support for Lakh, MAESTRO, and MIDIWorld datasets
- **Preprocessing**: music21-based MIDI normalization and feature extraction
- **Models**: LSTM and Transformer architectures for sequence prediction
- **Generation**: Temperature sampling for creative music synthesis
- **Deployment**: Flask and FastAPI APIs for inference
- **Advanced Features**: Multi-track generation and style transfer ready

## Architecture

### The Generative Pipeline

```
Raw MIDI Files
      ↓
[Step 1: Data Collection]
  - Lakh MIDI Dataset (180K+ files)
  - MAESTRO (Classical Piano)
  - MIDIWorld (Genre-specific)
      ↓
[Step 2: Preprocessing with music21]
  - Extract note pitches (C4, D#5, etc.)
  - Extract durations (quarter notes)
  - Extract offsets (timing)
  - Normalize and tokenize
      ↓
[Step 3: Build Vocabulary]
  - Create note-to-index mapping
  - Build training sequences (fixed window size)
      ↓
[Step 4: Train Model]
  - Input: Sequence of note indices
  - Embedding Layer
  - LSTM/Transformer Layers
  - Softmax Output (next note probability)
      ↓
[Step 5: Generate Music]
  - Feed seed sequence
  - Sample next note (with temperature)
  - Append & shift window
  - Repeat to build full composition
      ↓
Output: MIDI File
```

## Project Structure

```
music-generation/
├── src/
│   ├── data/                      # Data loading and MIDI utilities
│   │   ├── dataset_loader.py      # Download & load datasets
│   │   └── midi_utils.py          # Note conversion, vocabulary
│   ├── preprocessing/             # MIDI preprocessing
│   │   └── music21_preprocessor.py
│   ├── models/                    # Model architectures
│   │   ├── lstm_model.py          # TensorFlow & PyTorch LSTM
│   │   └── transformer_model.py   # Transformer variants
│   ├── training/                  # Training pipeline
│   │   └── trainer.py
│   ├── generation/                # Music generation
│   │   └── generator.py           # Temperature sampling
│   ├── visualization/             # Piano roll & plots
│   │   └── visualizer.py
│   └── deployment/                # API servers
│       ├── flask_app.py
│       └── fastapi_app.py
├── configs/
│   ├── config.py                  # Configuration classes
│   └── config.json                # Default settings
├── notebooks/                     # Jupyter notebooks
├── outputs/                       # Generated MIDI files
├── train.py                       # Training script
├── generate.py                    # Generation script
├── main.py                        # CLI interface
├── requirements.txt               # Dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- (Optional) NVIDIA GPU with CUDA for faster training

### Quick Start

1. **Clone and setup**
```bash
cd music-generation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Framework Selection

**TensorFlow (Recommended for beginners):**
```bash
pip install tensorflow>=2.12.0 keras>=2.12.0
```

**PyTorch (For advanced users):**
```bash
pip install torch>=2.0.0 torchaudio>=2.0.0
```

## Usage

### 1. Prepare Data

Download MIDI datasets:

**MAESTRO** (Classical Piano - Recommended)
```bash
# Download from: https://storage.googleapis.com/magenta-datasets/maestro/v3.0.0/maestro-v3.0.0.zip
# Extract to: ./data/maestro-v3.0.0
```

**Lakh MIDI Dataset**
```bash
# Visit: http://codespeaker.com/public/lakh/
# Provides 180K+ MIDI files
```

### 2. Train a Model

**Using LSTM (TensorFlow):**
```bash
python train.py --data_path ./data/maestro-v3.0.0
```

**Custom configuration:**
```bash
python train.py --data_path ./data --config configs/config.json
```

**Training parameters** (edit `configs/config.json`):
```json
{
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "model": {
    "model_type": "lstm"
  }
}
```

### 3. Generate Music

**Basic generation:**
```bash
python generate.py --length 200 --temperature 1.0
```

**With visualization:**
```bash
python generate.py --length 100 --temperature 0.8 --visualize
```

**Parameters:**
- `--length`: Number of notes to generate (default: 100)
- `--temperature`: Sampling temperature
  - `0.0`: Deterministic (always pick highest probability)
  - `1.0`: Standard sampling (most realistic)
  - `> 1.0`: More creative and random
- `--output`: Custom output filename
- `--visualize`: Create piano roll visualization

### 4. Deploy as API

**Flask Server:**
```bash
python main.py deploy --api flask --port 5000
```

Then access:
```bash
# Generate music
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"temperature": 1.0, "generate_length": 100}'

# List generated files
curl http://localhost:5000/api/files

# Download MIDI
curl http://localhost:5000/api/download/generated_0.mid -o output.mid
```

**FastAPI Server:**
```bash
python main.py deploy --api fastapi --port 8000
```

Interactive API docs at: `http://localhost:8000/docs`

## Model Architecture

### LSTM Model
```
Input (sequence) → Embedding → LSTM → LSTM → Dense → Softmax
                     128        256      256     512      vocab_size
```

- **Embedding Dimension**: 128
- **LSTM Units**: 256-512
- **Layers**: 2-3
- **Dropout**: 0.2 (prevents overfitting)
- **Output**: Probability distribution over next note

### Transformer Model

```
Input → Embedding → Positional Encoding
         ↓
    [Multi-Head Attention]
         ↓
    [Feed Forward]
         ↓
    [Repeat N times]
         ↓
    Dense → Softmax
```

- **Embedding**: 128-dimensional
- **Heads**: 8
- **Feed-forward**: 512-dimensional
- **Layers**: 4
- **Position Encoding**: Added to input embeddings

## Key Concepts

### Temperature Sampling
Instead of always picking the highest probability note:

```python
# Temperature sampling formula
prob_adjusted = exp(log(p) / temperature) / normalization

# Examples:
- temperature=0.0 → Always pick argmax (deterministic)
- temperature=1.0 → Use original probabilities (realistic)
- temperature=2.0 → Smooth distribution (more creative)
```

### Vocabulary Mapping
Every unique MIDI note gets an integer index:
```python
{
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
  60: 4,      # Middle C
  61: 5,      # C#
  ...
}
```

### Sequence Creation
Fixed-window sliding window for training:
```
Original: [60, 64, 67, 69, 72, 76, 79, ...]
          
Sequence 1: [60, 64, 67, 69, 72] → predict 76
Sequence 2: [64, 67, 69, 72, 76] → predict 79
...
```

## Advanced Features

### Multi-Track Generation
Train separate models for different instruments:
```python
# Future: Train on drums, bass, melody separately
# Then combine outputs for full arrangements
```

### Style Transfer
Blend model weights from different genres:
```python
# Classical model weights: 70%
# Jazz model weights: 30%
# Result: Classical-jazz fusion
```

### Real-Time Generation
Stream generation results for interactive applications:
```python
for note in generator.generate_sequence_streaming(seed, 1000):
    play_note(note)  # Play each note as it's generated
```

## Performance Tips

### Training
- **GPU**: Use NVIDIA GPU + CUDA for 10x+ speedup
- **Batch Size**: Larger batches (~64) for better GPU utilization
- **Data**: More diverse data = better generalization
- **Dropout**: Increase if overfitting occurs

### Inference
- **Batch Generation**: Generate multiple sequences in parallel
- **Model Caching**: Load model once, reuse for multiple generations
- **Quantization**: Compress model for faster inference (TensorFlow)

## Datasets

| Dataset | Size | Quality | Genre | URL |
|---------|------|---------|-------|-----|
| **MAESTRO** | 1,200 | High | Classical Piano | [Download](https://storage.googleapis.com/magenta-datasets/maestro/v3.0.0/maestro-v3.0.0.zip) |
| **Lakh MIDI** | 180K+ | Mixed | Various | [Homepage](http://codespeaker.com/public/lakh/) |
| **MIDIWorld** | 50K+ | Variable | Specific Genres | [Homepage](http://www.midiworld.com/) |

## Citation

If you use this project, please cite:

```bibtex
@software{musicgen2024,
  author = {Your Name},
  title = {Music Generation AI: Deep Learning for Musical Composition},
  year = {2024},
  url = {https://github.com/yourusername/music-generation}
}
```

## References

### Key Papers
- [Music Transformer](https://arxiv.org/abs/1809.04281) - Huang et al. (2018)
- [MuseNet](https://openai.com/blog/musenet/) - OpenAI (2019)
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) - Hawthorne et al. (2018)

### Libraries
- [music21](http://web.mit.edu/music21/) - Music analysis and generation
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [librosa](https://librosa.org/) - Audio analysis

## Troubleshooting

### Common Issues

**No MIDI files found**
```
Solution: Download datasets and place in ./data/
Check configs/config.json for correct data_dir path
```

**Model not found**
```
Solution: Train model first with: python train.py --data_path ./data
Check models/ directory exists
```

**Out of memory**
```
Solution: Reduce batch_size in configs/config.json
Use fewer MIDI files (set max_files in config)
```

**slow training**
```
Solution: Check if GPU is being used
Install CUDA/cuDNN for TensorFlow/PyTorch
Reduce sequence_length or batch_size
```

## Roadmap

- [ ] Web UI for interactive generation
- [ ] Real-time MIDI playback
- [ ] Multi-track generation
- [ ] Style transfer pipeline
- [ ] Reinforcement learning for music quality
- [ ] MIDI to audio synthesis (with Synthesizer)
- [ ] Pre-trained model zoo
- [ ] Docker deployment

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [link]
- Documentation: [link]
- Email: [your-email]

---

**Made with ❤️ for musicians and AI enthusiasts**

Built at FAST-NUCES for AI project research.
