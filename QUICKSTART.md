# Quick Start Guide

## For FAST-NUCES AI Project

This guide will get you up and running with music generation in 5 minutes.

## Step 1: Install

```bash
pip install -r requirements.txt
```

## Step 2: Download Sample Data

Visit one of these and download sample MIDI files:
- **MAESTRO** (recommended): https://storage.googleapis.com/magenta-datasets/maestro/v3.0.0/maestro-v3.0.0.zip
- Extract to: `./data/maestro-v3.0.0` (or any directory)

## Step 3: Train Model

```bash
python train.py --data_path ./data/maestro-v3.0.0
```

This will:
- Load all MIDI files
- Preprocess them
- Train an LSTM model
- Save model to `./models/`

Training time: 30 minutes to 2 hours (depends on GPU and dataset size)

## Step 4: Generate Music

```bash
python generate.py --length 150 --temperature 1.0 --visualize
```

Output MIDI file will be saved to `./outputs/`

## Step 5: Listen!

Play the generated MIDI file with any MIDI player (e.g., MuseScore, GarageBand, Piano)

## What's Next?

- Explore `notebooks/` for detailed examples
- Try different temperatures: `--temperature 0.5` (realistic) to `2.0` (creative)
- Deploy as API: `python main.py deploy --api flask`
- Experiment with Transformer model (edit `configs/config.json`)

## Common Commands

```bash
# Train with custom config
python train.py --data_path ./data --config configs/config.json

# Generate with different settings
python generate.py --length 200 --temperature 0.7 --output my_composition.mid

# Deploy Flask API
python main.py deploy --api flask --port 5000

# Deploy FastAPI
python main.py deploy --api fastapi --port 8000
```

## Troubleshooting

**Error: No MIDI files found**
- Check data path is correct
- Ensure MIDI files end with `.mid`

**Out of memory during training**
- Reduce `batch_size` in `configs/config.json`
- Use fewer MIDI files

**Slow training**
- Check GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Or `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

**Audio quality issues**
- Train on more data
- Increase temperature during generation
- Try 50+ epochs of training

## Project Structure

- `train.py` - Training script
- `generate.py` - Generation script
- `src/` - Core library modules
- `configs/` - Configuration files
- `outputs/` - Generated MIDI files
- `models/` - Saved trained models
