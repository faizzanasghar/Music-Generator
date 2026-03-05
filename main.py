"""Main entry point for music generation project."""

import argparse
import logging
from pathlib import Path
import sys

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Music Generation AI - Generate music using deep learning"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a music generation model")
    train_parser.add_argument("--data_path", type=str, help="Path to MIDI files")
    train_parser.add_argument("--config", type=str, help="Configuration file")
    train_parser.set_defaults(func=run_train)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate music from trained model")
    gen_parser.add_argument("--model_type", type=str, choices=['lstm', 'transformer'])
    gen_parser.add_argument("--framework", type=str, choices=['tensorflow', 'pytorch'])
    gen_parser.add_argument("--length", type=int, help="Sequence length to generate")
    gen_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    gen_parser.add_argument("--output", type=str, help="Output filename")
    gen_parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    gen_parser.add_argument("--config", type=str, help="Configuration file")
    gen_parser.set_defaults(func=run_generate)
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy API server")
    deploy_parser.add_argument("--api", type=str, choices=['flask', 'fastapi'],
                             default='flask', help="API framework")
    deploy_parser.add_argument("--port", type=int, help="Server port")
    deploy_parser.add_argument("--config", type=str, help="Configuration file")
    deploy_parser.set_defaults(func=run_deploy)
    
    # Version
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


def run_train(args):
    """Run training pipeline."""
    import train as train_module
    train_module.main(args)


def run_generate(args):
    """Run generation pipeline."""
    import generate as gen_module
    gen_module.main(args)


def run_deploy(args):
    """Run deployment."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    if args.api == 'flask':
        from src.deployment.flask_app import create_flask_app
        from src.data.midi_utils import MidiUtils
        from src.models.lstm_model import LSTMMusic
        from src.generation.generator import MusicGenerator
        
        # Load model and vocab
        models_dir = Path("./models")
        vocab_path = models_dir / "vocabulary.json"
        model_path = models_dir / "lstm_tensorflow.h5"
        
        if not vocab_path.exists() or not model_path.exists():
            logger.error("Model or vocabulary not found. Please train first.")
            return
        
        vocab = MidiUtils.load_vocabulary(str(vocab_path))
        model = LSTMMusic(vocab_size=vocab['vocab_size'])
        model.load(str(model_path))
        
        generator = MusicGenerator(model.get_model(), vocab)
        app = create_flask_app(model.get_model(), generator, vocab)
        
        port = args.port or 5000
        logger.info(f"Starting Flask server on http://localhost:{port}")
        app.run(debug=True, port=port)
    
    elif args.api == 'fastapi':
        import uvicorn
        from src.deployment.fastapi_app import create_fastapi_app
        from src.data.midi_utils import MidiUtils
        from src.models.lstm_model import LSTMMusic
        from src.generation.generator import MusicGenerator
        
        # Load model and vocab
        models_dir = Path("./models")
        vocab_path = models_dir / "vocabulary.json"
        model_path = models_dir / "lstm_tensorflow.h5"
        
        if not vocab_path.exists() or not model_path.exists():
            logger.error("Model or vocabulary not found. Please train first.")
            return
        
        vocab = MidiUtils.load_vocabulary(str(vocab_path))
        model = LSTMMusic(vocab_size=vocab['vocab_size'])
        model.load(str(model_path))
        
        generator = MusicGenerator(model.get_model(), vocab)
        app = create_fastapi_app(model.get_model(), generator, vocab)
        
        port = args.port or 8000
        logger.info(f"Starting FastAPI server on http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
