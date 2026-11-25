"""FACE-C application configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
ENCODER_PATH = MODEL_DIR / "encoder.keras"
DECODER_PATH = MODEL_DIR / "decoder.keras"
OUTPUT_DIR = ROOT_DIR / "outputs"
IMAGE_SIZE = (128, 128)
EXPECTED_LATENT_SHAPE = (16, 16, 32)
MAX_HEADER_BYTES = 1_000_000
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
