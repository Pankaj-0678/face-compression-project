"""FACE-C Streamlit demonstration application."""
from __future__ import annotations
import io
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
from config import ENCODER_PATH, DECODER_PATH, OUTPUT_DIR, EXPECTED_LATENT_SHAPE
from src.codec import compress_image, decompress_image
from src.facec_format import read_facec_file
from src.metrics import compression_metrics, image_quality
from src.model_loader import load_models
from src.preprocessing import image_to_uint8
from src.utils import format_bytes, png_bytes

st.set_page_config(page_title="FACE-C — Neural Face Compression", page_icon="🧑", layout="wide")
st.markdown("""<style>
.stApp{background:#f5f7fb;color:#172033}.block-container{padding-top:2rem;max-width:1200px}
[data-testid="stMetric"]{background:white;padding:1rem;border-radius:14px;border:1px solid #e5eaf2}
.hero{padding:2rem;border-radius:20px;background:linear-gradient(120deg,#12213c,#275b8c);color:white;margin-bottom:1rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained encoder and decoder…")
def models():
    return load_models(ENCODER_PATH, DECODER_PATH)


def get_models():
    try: return models()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Trained models are unavailable or incompatible: {exc}")
        return None


def show_metrics(stats: dict):
    labels = [("Original RGB image", "original_image_size"), ("Original float32 latent", "original_float32_latent_size"),
              ("Quantized latent", "quantized_latent_size"), ("Huffman payload", "huffman_payload_size"),
              ("Complete FACE-C file", "facec_file_size")]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, labels): col.metric(label, format_bytes(stats.get(key)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Latent compression ratio", f"{stats['latent_compression_ratio']:.2f}×" if stats.get('latent_compression_ratio') else "—")
    c2.metric("Raw image compression ratio", f"{stats['raw_image_compression_ratio']:.2f}×" if stats.get('raw_image_compression_ratio') else "—")
    c3.metric("Storage reduction", f"{stats['storage_reduction_percent']:.1f}%" if stats.get('storage_reduction_percent') is not None else "—")


st.sidebar.title("FACE-C")
page = st.sidebar.radio("Navigate", ["Dashboard", "Compress Image", "Decompress Image", "Image Comparison"])
st.sidebar.caption("CNN autoencoder · 8-bit min-max quantization · Huffman coding")

if page == "Dashboard":
    st.markdown('<div class="hero"><h1>FACE-C — Neural Face Compression</h1><p>Compress face images into a self-contained binary format, then reconstruct them with trained CNN autoencoder models.</p></div>', unsafe_allow_html=True)
    st.subheader("Project objective")
    st.write("Demonstrate a learned image codec: an encoder maps a 128 × 128 RGB image to a compact latent tensor; uniform quantization and Huffman coding store that representation in a `.facec` file. The decoder reconstructs an image from the file alone.")
    left, right = st.columns(2)
    with left:
        st.markdown("### Compression pipeline")
        st.code("RGB image → resize/normalize → trained CNN encoder → 16×16×32 latent → uint8 min-max quantization → Huffman → FAC1 file", language="text")
    with right:
        st.markdown("### Decompression pipeline")
        st.code("FAC1 file → validate header → Huffman decode → dequantize → trained CNN decoder → reconstructed RGB image", language="text")
    st.markdown("### Model specifications")
    latent_description = f"{EXPECTED_LATENT_SHAPE[0]} × {EXPECTED_LATENT_SHAPE[1]} × {EXPECTED_LATENT_SHAPE[2]} ({np.prod(EXPECTED_LATENT_SHAPE):,} values)"
    st.table({"Component": ["Encoder input", "Encoder output / decoder input", "Decoder output", "Quantizer"], "Specification": ["128 × 128 × 3 RGB, [0,1]", latent_description, "128 × 128 × 3 RGB", "Per-image min-max, 8-bit uint8"]})
    st.info("Quantization is lossy. Huffman coding is lossless for the quantized symbols. Quality metrics require both original and reconstructed images.")

elif page == "Compress Image":
    st.title("Compress an image")
    upload = st.file_uploader("Choose JPG, PNG, or WebP", type=["jpg", "jpeg", "png", "webp"], key="compress")
    if upload:
        original = Image.open(upload).convert("RGB")
        st.image(original, caption=f"Original · {original.width} × {original.height}", width=360)
        loaded = get_models()
        if loaded and st.button("Compress image", type="primary"):
            encoder, _, _ = loaded
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUTPUT_DIR / (Path(upload.name).stem + ".facec")
            with st.spinner("Encoding and writing FACE-C file…"):
                result = compress_image(original, encoder, out)
            stats = compression_metrics(result["original_image_size"], result["original_latent_size"], result["quantized_latent_size"], result["huffman_payload_size"], result["facec_file_size"], len(upload.getvalue()))
            show_metrics(stats)
            st.write(f"Latent shape: `{result['latent'].shape}` · Quantization range: `{result['metadata']['latent_min']:.7g}` to `{result['metadata']['latent_max']:.7g}` · Huffman bits: `{result['bit_length']:,}`")
            st.download_button("Download .facec", out.read_bytes(), file_name=out.name, mime="application/octet-stream")

elif page == "Decompress Image":
    st.title("Decompress a FACE-C file")
    upload = st.file_uploader("Choose a .facec file", type=["facec"])
    if upload:
        try:
            metadata, quantized = read_facec_file(upload.getvalue())
            st.success(f"Valid FACE-C v{metadata['version']} · latent shape {tuple(metadata['latent_shape'])} · {quantized.size:,} symbols")
            loaded = get_models()
            if loaded and st.button("Reconstruct image", type="primary"):
                _, decoder, _ = loaded
                with st.spinner("Decoding latent representation…"):
                    result = decompress_image(upload.getvalue(), decoder)
                st.image(result["image"], caption="Reconstructed image", width=512)
                st.download_button("Download reconstructed PNG", png_bytes(result["image"]), file_name="reconstructed.png", mime="image/png")
                st.json({k: v for k, v in metadata.items() if k != "codebook"} | {"codebook_symbols": len(metadata["codebook"]), "file_size_bytes": len(upload.getvalue())})
        except (ValueError, OSError) as exc: st.error(f"Could not read this FACE-C file: {exc}")

else:
    st.title("Image comparison")
    st.write("Provide the original image and either its FACE-C file or a reconstructed image. PSNR and SSIM are only shown when both images are available.")
    col_a, col_b = st.columns(2)
    with col_a: orig_upload = st.file_uploader("Original image", type=["jpg", "jpeg", "png", "webp"], key="compare_original")
    with col_b:
        face_upload = st.file_uploader("Reconstructed image or FACE-C", type=["jpg", "jpeg", "png", "webp", "facec"], key="compare_recon")
    if orig_upload and face_upload:
        original = Image.open(orig_upload).convert("RGB")
        try:
            if face_upload.name.lower().endswith(".facec"):
                loaded = get_models()
                if not loaded: st.stop()
                reconstructed = decompress_image(face_upload.getvalue(), loaded[1])["image"]
                face_size = len(face_upload.getvalue())
            else:
                reconstructed = np.asarray(Image.open(face_upload).convert("RGB"), dtype=np.float32) / 255.0
                face_size = len(face_upload.getvalue())
            orig_arr = image_to_uint8(original, (128, 128)).astype(np.float32) / 255.0
            recon_arr = image_to_uint8(reconstructed, (128, 128)).astype(np.float32) / 255.0
            a, b = st.columns(2); a.image(original, caption="Original", use_container_width=True); b.image(reconstructed, caption="Reconstructed", use_container_width=True)
            quality = image_quality(orig_arr, recon_arr)
            m1, m2 = st.columns(2); m1.metric("PSNR", f"{quality['psnr']:.2f} dB"); m2.metric("SSIM", f"{quality['ssim']:.4f}")
            st.write(f"Original RGB storage at 128×128: {format_bytes(128*128*3)} · supplied original file: {format_bytes(len(orig_upload.getvalue()))} · comparison file: {format_bytes(face_size)}")
            if face_upload.name.lower().endswith(".facec"):
                st.write(f"Raw image compression ratio: {(128*128*3)/face_size:.2f}×")
        except (ValueError, OSError) as exc: st.error(f"Comparison failed: {exc}")
