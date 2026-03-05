"""Flask-based deployment for music generation."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FlaskApp:
    """Flask application for music generation API."""

    def __init__(self, model, generator, vocab, output_dir: str = "outputs"):
        """
        Initialize Flask app.
        
        Args:
            model: Trained model
            generator: MusicGenerator instance
            vocab: Vocabulary dictionary
            output_dir: Directory for generated MIDI files
        """
        self.model = model
        self.generator = generator
        self.vocab = vocab
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_app(self):
        """Create and configure Flask app."""
        try:
            from flask import Flask, jsonify, request
        except ImportError:
            raise ImportError("Flask is required. Install with: pip install flask")

        app = Flask(__name__)

        @app.route('/')
        def home():
            """Home endpoint."""
            return jsonify({"message": "Music Generation API", "version": "1.0"})

        @app.route('/api/generate', methods=['POST'])
        def generate_music():
            """
            Generate music endpoint.
            
            POST data:
            {
                "seed_length": 50,
                "generate_length": 100,
                "temperature": 1.0
            }
            """
            try:
                data = request.get_json()
                
                seed_length = data.get('seed_length', 50)
                generate_length = data.get('generate_length', 100)
                temperature = data.get('temperature', 1.0)

                logger.info(f"Generating music: seed={seed_length}, length={generate_length}, temp={temperature}")

                # Get random seed from validation data
                import numpy as np
                seed = list(np.random.randint(1, self.vocab['vocab_size'], size=seed_length))

                # Generate
                sequence = self.generator.generate_sequence(seed, generate_length, temperature)

                # Save MIDI
                filename = f"generated_{len(os.listdir(self.output_dir))}.mid"
                filepath = self.output_dir / filename

                self.generator.generate_midi(sequence, str(filepath))

                return jsonify({
                    "success": True,
                    "filename": filename,
                    "filepath": str(filepath),
                    "sequence_length": len(sequence)
                })

            except Exception as e:
                logger.error(f"Error generating music: {e}")
                return jsonify({"success": False, "error": str(e)}), 400

        @app.route('/api/download/<filename>', methods=['GET'])
        def download_file(filename: str):
            """Download generated MIDI file."""
            try:
                from flask import send_file
                filepath = self.output_dir / filename

                if not filepath.exists():
                    return jsonify({"error": "File not found"}), 404

                return send_file(filepath, as_attachment=True)

            except Exception as e:
                logger.error(f"Error downloading file: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/files', methods=['GET'])
        def list_files():
            """List all generated files."""
            files = list(self.output_dir.glob("*.mid"))
            return jsonify({"files": [f.name for f in files]})

        @app.route('/api/info', methods=['GET'])
        def get_info():
            """Get model and generator info."""
            return jsonify({
                "vocab_size": self.vocab['vocab_size'],
                "generated_files": len(list(self.output_dir.glob("*.mid")))
            })

        return app


# Example usage function
def create_flask_app(model, generator, vocab, output_dir: str = "outputs"):
    """
    Create Flask application for music generation.
    
    Usage:
        from src.deployment.flask_app import create_flask_app
        app = create_flask_app(model, generator, vocab)
        app.run(debug=True, port=5000)
    """
    flask_app = FlaskApp(model, generator, vocab, output_dir)
    return flask_app.create_app()
