"""FastAPI-based deployment for music generation."""

import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    """Request model for music generation."""
    seed_length: int = 50
    generate_length: int = 100
    temperature: float = 1.0


class FastAPIApp:
    """FastAPI application for music generation API."""

    def __init__(self, model, generator, vocab, output_dir: str = "outputs"):
        """
        Initialize FastAPI app.
        
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
        """Create and configure FastAPI app."""
        try:
            from fastapi import FastAPI, UploadFile
            from fastapi.responses import FileResponse, JSONResponse
            from fastapi.staticfiles import StaticFiles
        except ImportError:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

        app = FastAPI(
            title="Music Generation API",
            description="AI-powered music generation using LSTM/Transformer models",
            version="1.0"
        )

        @app.get("/")
        async def home():
            """Home endpoint."""
            return {"message": "Music Generation API", "version": "1.0"}

        @app.post("/api/generate")
        async def generate_music(request: GenerationRequest):
            """
            Generate music endpoint.
            
            Args:
                request: GenerationRequest with parameters
                
            Returns:
                Generated MIDI file info
            """
            try:
                import numpy as np
                import os

                logger.info(f"Generating music: seed={request.seed_length}, "
                           f"length={request.generate_length}, temp={request.temperature}")

                # Create seed
                seed = list(np.random.randint(1, self.vocab['vocab_size'], 
                                              size=request.seed_length))

                # Generate
                sequence = self.generator.generate_sequence(
                    seed, request.generate_length, request.temperature
                )

                # Save MIDI
                filename = f"generated_{len(os.listdir(self.output_dir))}.mid"
                filepath = self.output_dir / filename

                self.generator.generate_midi(sequence, str(filepath))

                return {
                    "success": True,
                    "filename": filename,
                    "filepath": str(filepath),
                    "sequence_length": len(sequence)
                }

            except Exception as e:
                logger.error(f"Error generating music: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": str(e)}
                )

        @app.get("/api/download/{filename}")
        async def download_file(filename: str):
            """Download generated MIDI file."""
            try:
                filepath = self.output_dir / filename

                if not filepath.exists():
                    return JSONResponse(
                        status_code=404,
                        content={"error": "File not found"}
                    )

                return FileResponse(filepath, filename=filename)

            except Exception as e:
                logger.error(f"Error downloading file: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )

        @app.get("/api/files")
        async def list_files():
            """List all generated files."""
            files = list(self.output_dir.glob("*.mid"))
            return {"files": [f.name for f in files]}

        @app.get("/api/info")
        async def get_info():
            """Get model and generator info."""
            return {
                "vocab_size": self.vocab['vocab_size'],
                "generated_files": len(list(self.output_dir.glob("*.mid")))
            }

        return app


# Example usage function
def create_fastapi_app(model, generator, vocab, output_dir: str = "outputs"):
    """
    Create FastAPI application for music generation.
    
    Usage:
        from src.deployment.fastapi_app import create_fastapi_app
        app = create_fastapi_app(model, generator, vocab)
        # Run with: uvicorn src.deployment.fastapi_app:app --reload
    """
    fastapi_app = FastAPIApp(model, generator, vocab, output_dir)
    return fastapi_app.create_app()
