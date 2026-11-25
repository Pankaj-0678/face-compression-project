# FACE-C — Neural Face Compression

FACE-C is a modular Streamlit demonstration of face image compression with a trained convolutional autoencoder. It does not train models or synthesize predictions: provide the project's actual Keras encoder and decoder to use the application.

## Architecture and algorithm

Compression converts an input image to RGB, resizes it to 128×128, and normalizes pixels to [0,1]. The configured trained encoder produces a batch output of shape `(1, 16, 16, 32)`. Per-image min-max uniform quantization maps those 8,192 float values to uint8. A hand-built Huffman coder compresses the symbols losslessly, and the result is written to `.facec`.

Decompression validates the file and decodes its Huffman payload to the exact original quantized uint8 symbols. It reconstructs float latent values from the stored min/max, then passes the latent tensor to the trained decoder. Quantization can introduce small image differences; Huffman coding itself is lossless.

The v1 binary layout is `FAC1` (4 bytes), a big-endian unsigned 32-bit JSON header length, UTF-8 JSON metadata, then the packed Huffman payload. Metadata contains version, latent shape, min/max bounds, Huffman bit length, symbol-to-bit-string codebook, and a SHA-256 payload integrity checksum. Padding bits are excluded by the stored bit length. Header and decoded-size limits are enforced before allocation.

## Structure

```text
app.py, config.py, requirements.txt
models/                 # user-provided encoder.keras and decoder.keras
src/                    # preprocessing, model loading, quantization, Huffman, file format, codec, metrics
tests/                  # algorithm-only tests and lightweight codec tests
outputs/                # generated FACE-C files
```

## Install and run

Use Python 3.10 or newer, then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Place the trained models at `models/encoder.keras` and `models/decoder.keras`, or change `ENCODER_PATH` and `DECODER_PATH` in `config.py`. The app loads each model once through Streamlit resource caching and reports exact runtime shape mismatches.

Run algorithm tests with `pytest -q`. These tests use synthetic byte arrays and small deterministic test doubles; they do not train models and do not claim real-model reconstruction quality. Launch the UI with:

```bash
streamlit run app.py
```

The dashboard is usable before models are added. Compression, decompression, and FACE-C comparison require the trained files. Tests verify exact equality of the quantized symbols before/after Huffman coding and a synthetic codec round-trip.

## Metrics

Latent compression ratio is float32 latent bytes divided by complete `.facec` bytes. Raw image compression ratio is uncompressed 128×128 RGB bytes divided by complete `.facec` bytes. Storage reduction uses that same raw RGB baseline. PSNR and SSIM are shown only when an original is provided, with both images compared as RGB float values in [0,1] using a fixed data range of 1.
