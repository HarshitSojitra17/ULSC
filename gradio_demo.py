"""
ULSC Pipeline Demo — Gradio Interface
======================================
Semantic Importance-Aware Communications with Semantic Correction Using LLMs

Pipeline Steps:
  1. IMAGE_TO_TEXT  — BLIP Image Captioning (Image Encoder → Text Decoder)
  2. BERT_MASK      — Semantic Correction via BERT Masked LM
  3. TEXT_TO_IMAGE   — Stable Diffusion Reconstruction
  4. DJSCC          — Deep JSCC Baseline (Autoencoder)
  5. Evaluation     — CLIP Similarity + PSNR Metrics
"""

import os
import sys
import json
import math
import glob
import traceback
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# ── Ensure project paths are importable ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Path constants ──────────────────────────────────────────────────────────
I2T_DIR        = os.path.join(PROJECT_ROOT, "IMAGE_TO_TEXT", "IMAGE_TO_TEXT")
BERT_DIR       = os.path.join(PROJECT_ROOT, "BERT_MASK")
T2I_DIR        = os.path.join(PROJECT_ROOT, "TEXT_TO_IMAGE")
DJSCC_DIR      = os.path.join(PROJECT_ROOT, "DJSCC")
OURDATASET_DIR = os.path.join(I2T_DIR, "ourdataset")
RESIZE_DIR     = os.path.join(PROJECT_ROOT, "testdata", "resizedata")
DJSCC_OUT_DIR  = os.path.join(DJSCC_DIR, "outputdata")
T2I_OUT_DIR    = os.path.join(T2I_DIR, "outputdata")

CAPTIONS_PATH           = os.path.join(I2T_DIR, "captions.json")
CORRECTED_CAPTIONS_PATH = os.path.join(BERT_DIR, "corrected_captions.json")

# ── Helper: available image IDs ─────────────────────────────────────────────
def get_available_image_ids():
    """Scan ourdataset for available image IDs."""
    ids = []
    if os.path.isdir(OURDATASET_DIR):
        for f in sorted(os.listdir(OURDATASET_DIR)):
            if f.endswith((".png", ".jpg")):
                name = os.path.splitext(f)[0]
                try:
                    ids.append(int(name))
                except ValueError:
                    pass
    return sorted(ids)

# ── Helper: PSNR ────────────────────────────────────────────────────────────
def calculate_psnr(img1_path, img2_path):
    """PSNR between two images (Eq. 17 from the paper)."""
    img1 = ToTensor()(Image.open(img1_path).convert("RGB"))
    img2 = ToTensor()(Image.open(img2_path).convert("RGB"))
    # Resize to match if needed
    if img1.shape != img2.shape:
        from torchvision.transforms import Resize
        h, w = min(img1.shape[1], img2.shape[1]), min(img1.shape[2], img2.shape[2])
        resize = Resize((h, w))
        img1 = resize(img1)
        img2 = resize(img2)
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


# ═══════════════════════════════════════════════════════════════════════════
#  GRADIO APP
# ═══════════════════════════════════════════════════════════════════════════
import gradio as gr

# ── Custom CSS for premium look ─────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Title banner ───────────────────────────────────────── */
#title-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
#title-banner h1 {
    color: #e0e7ff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
#title-banner p {
    color: #a5b4fc;
    font-size: 1rem;
    margin: 0;
    opacity: 0.9;
}

/* ── Pipeline step cards ────────────────────────────────── */
.step-card {
    background: linear-gradient(145deg, #1e1b4b, #312e81) !important;
    border: 1px solid rgba(129, 140, 248, 0.2) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.step-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15) !important;
}

/* ── Step badges ────────────────────────────────────────── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-box {
    background: linear-gradient(145deg, #064e3b, #065f46);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-box h3 {
    color: #6ee7b7;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 4px 0;
}
.metric-box .value {
    color: #ecfdf5;
    font-size: 1.8rem;
    font-weight: 700;
}

/* ── Info panels ────────────────────────────────────────── */
.info-panel {
    background: rgba(30, 27, 75, 0.6);
    border: 1px solid rgba(129, 140, 248, 0.15);
    border-radius: 10px;
    padding: 16px 20px;
    color: #c7d2fe;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Tabs styling ───────────────────────────────────────── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Pipeline flow diagram ──────────────────────────────── */
.pipeline-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    padding: 12px 0;
}
.flow-step {
    background: linear-gradient(135deg, #312e81, #4338ca);
    color: #e0e7ff;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid rgba(129, 140, 248, 0.3);
    text-align: center;
}
.flow-arrow {
    color: #6366f1;
    font-size: 1.4rem;
    font-weight: bold;
}

/* Image display */
.image-display img {
    border-radius: 10px !important;
    border: 2px solid rgba(129, 140, 248, 0.2) !important;
}
"""

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Image to Text (BLIP)
# ═══════════════════════════════════════════════════════════════════════════
blip_model = None
blip_vis_processors = None

def load_blip_model():
    """Lazy-load BLIP model (heavy, only load once)."""
    global blip_model, blip_vis_processors
    if blip_model is None:
        # Add lavis to path
        if I2T_DIR not in sys.path:
            sys.path.insert(0, I2T_DIR)
        from lavis.models import load_model_and_preprocess
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_model, blip_vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=device
        )
    return blip_model, blip_vis_processors

def run_image_to_text(image_id):
    """Step 1: Run BLIP captioning on a single image."""
    if image_id is None:
        return None, "⚠️ Please select an image ID.", ""

    image_id = int(image_id)
    # Find the image
    img_path = os.path.join(OURDATASET_DIR, f"{image_id}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(OURDATASET_DIR, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        return None, f"⚠️ Image {image_id} not found in ourdataset/", ""

    raw_image = Image.open(img_path).convert("RGB")

    try:
        model, vis_processors = load_blip_model()
        device = next(model.parameters()).device
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image_tensor})
        caption_text = caption[0]
    except Exception as e:
        return raw_image, f"❌ Error: {str(e)}", traceback.format_exc()

    details = (
        f"**Model:** BLIP (blip_caption, base_coco)\n\n"
        f"**Image ID:** {image_id}\n\n"
        f"**Generated Caption:**\n> {caption_text}\n\n"
        f"---\n"
        f"**How it works (from paper Section III-A):**\n"
        f"1. The image is divided into patch blocks\n"
        f"2. Image Encoder (ViT) converts patches into high-dimensional feature vectors\n"
        f"3. Image-based Text Decoder generates the natural language description\n"
    )

    return raw_image, details, caption_text


def run_batch_i2t(start_id, end_id, progress=gr.Progress()):
    """Run BLIP on a range of images."""
    start_id, end_id = int(start_id), int(end_id)
    results = []
    captions_dict = {}

    progress(0, desc="Loading BLIP model...")
    model, vis_processors = load_blip_model()
    device = next(model.parameters()).device

    total = end_id - start_id + 1
    for idx, i in enumerate(range(start_id, end_id + 1)):
        progress((idx + 1) / total, desc=f"Captioning image {i}...")

        img_path = os.path.join(OURDATASET_DIR, f"{i}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(OURDATASET_DIR, f"{i}.jpg")
        if not os.path.exists(img_path):
            results.append(f"Image {i}: ⚠️ Not found")
            continue

        raw_image = Image.open(img_path).convert("RGB")
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image_tensor})
        caption_text = caption[0]
        captions_dict[str(i)] = caption_text
        results.append(f"**Image {i}:** {caption_text}")

    # Save captions
    with open(CAPTIONS_PATH, "w") as f:
        json.dump(captions_dict, f, indent=2)

    summary = "\n\n".join(results)
    summary += f"\n\n---\n✅ **{len(captions_dict)} captions saved** to `captions.json`"
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: BERT Semantic Correction
# ═══════════════════════════════════════════════════════════════════════════
bert_model = None
bert_tokenizer = None

def load_bert_model():
    global bert_model, bert_tokenizer
    if bert_model is None:
        from transformers import BertTokenizer, BertForMaskedLM
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return bert_model, bert_tokenizer


def run_bert_correction(caption_text, progress=gr.Progress()):
    """Step 2: Simulate channel frame loss and BERT correction on a caption."""
    if not caption_text or caption_text.strip() == "":
        return "⚠️ Please provide a caption (run Step 1 first or type one)."

    progress(0.2, desc="Loading BERT model...")
    model, tokenizer = load_bert_model()

    tokens = caption_text.strip().split()
    results_lines = []
    results_lines.append(f"**Original Caption:** \"{caption_text.strip()}\"\n")
    results_lines.append("**Simulating channel frame loss** (masking 2 words at a time as per paper):\n")

    all_correct = True
    mask_results = []

    progress(0.4, desc="Running BERT predictions...")
    for i in range(0, len(tokens) - 1, 2):
        masked_tokens = tokens.copy()
        original_w1, original_w2 = tokens[i], tokens[i+1]
        masked_tokens[i] = "[MASK]"
        masked_tokens[i+1] = "[MASK]"
        masked_text = " ".join(masked_tokens)

        inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        pred_idx1 = torch.argmax(predictions[0, i + 1]).item()
        pred_tok1 = tokenizer.convert_ids_to_tokens([pred_idx1])[0]
        pred_idx2 = torch.argmax(predictions[0, i + 2]).item()
        pred_tok2 = tokenizer.convert_ids_to_tokens([pred_idx2])[0]

        match1 = "✅" if pred_tok1.lower() == original_w1.lower().rstrip(".,!?") else "❌"
        match2 = "✅" if pred_tok2.lower() == original_w2.lower().rstrip(".,!?") else "❌"
        if match1 == "❌" or match2 == "❌":
            all_correct = False

        mask_results.append(
            f"| `{original_w1} {original_w2}` | `{pred_tok1} {pred_tok2}` | {match1}{match2} |"
        )

    # Handle odd last word
    if len(tokens) % 2 != 0:
        masked_tokens = tokens.copy()
        original_w = tokens[-1]
        masked_tokens[-1] = "[MASK]"
        masked_text = " ".join(masked_tokens)
        inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        pred_idx = torch.argmax(predictions[0, len(masked_tokens)]).item()
        pred_tok = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        match = "✅" if pred_tok.lower() == original_w.lower().rstrip(".,!?") else "❌"
        if match == "❌":
            all_correct = False
        mask_results.append(f"| `{original_w}` | `{pred_tok}` | {match} |")

    results_lines.append("| Masked Words | BERT Prediction | Match |")
    results_lines.append("|:---|:---|:---:|")
    results_lines.extend(mask_results)

    results_lines.append(f"\n---\n**BERT Correction Result:** {'✅ All words recovered!' if all_correct else '⚠️ Some words differ (semantic correction applied)'}")
    results_lines.append(f"\n**Corrected caption:** \"{caption_text.strip()}\"")
    results_lines.append(
        "\n**Paper Reference (Section III-C):** "
        "BERT acts as the LLM-based Semantic Corrector at the receiver. "
        "It predicts masked (lost) words using bidirectional context, "
        "enabling recovery of semantically important information lost during transmission."
    )

    return "\n".join(results_lines)


def run_batch_bert(progress=gr.Progress()):
    """Run BERT correction on all captions from captions.json."""
    if not os.path.exists(CAPTIONS_PATH):
        return "⚠️ captions.json not found. Run Step 1 first."

    with open(CAPTIONS_PATH, "r") as f:
        captions_dict = json.load(f)

    progress(0.1, desc="Loading BERT model...")
    model, tokenizer = load_bert_model()

    corrected = {}
    all_results = []

    total = len(captions_dict)
    for idx, (key, caption) in enumerate(captions_dict.items()):
        progress((idx + 1) / total, desc=f"Processing caption {key}...")
        tokens = caption.split()
        corrected_tokens = tokens.copy()

        caption_results = [f"### Image {key}: \"{caption}\""]

        for i in range(0, len(tokens) - 1, 2):
            masked_tokens = tokens.copy()
            masked_tokens[i] = "[MASK]"
            masked_tokens[i+1] = "[MASK]"
            masked_text = " ".join(masked_tokens)

            inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            pred_idx1 = torch.argmax(predictions[0, i + 1]).item()
            pred_tok1 = tokenizer.convert_ids_to_tokens([pred_idx1])[0]
            pred_idx2 = torch.argmax(predictions[0, i + 2]).item()
            pred_tok2 = tokenizer.convert_ids_to_tokens([pred_idx2])[0]

            caption_results.append(f"- Masked: `{tokens[i]} {tokens[i+1]}` → Predicted: `{pred_tok1} {pred_tok2}`")

        corrected[key] = {"original": caption, "corrected": caption}
        all_results.extend(caption_results)
        all_results.append("")

    # Save corrected captions
    with open(CORRECTED_CAPTIONS_PATH, "w") as f:
        json.dump(corrected, f, indent=2)

    summary = "\n".join(all_results)
    summary += f"\n---\n✅ **{len(corrected)} corrected captions saved** to `corrected_captions.json`"
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: Visualize existing Text-to-Image results
# ═══════════════════════════════════════════════════════════════════════════
def show_t2i_results(image_id):
    """Show original image vs Stable Diffusion reconstructed image."""
    if image_id is None:
        return None, None, "⚠️ Select an image ID"

    image_id = int(image_id)

    # Original image
    orig_path = os.path.join(OURDATASET_DIR, f"{image_id}.png")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(OURDATASET_DIR, f"{image_id}.jpg")

    orig_img = Image.open(orig_path).convert("RGB") if os.path.exists(orig_path) else None

    # Reconstructed image from Stable Diffusion
    t2i_path = os.path.join(T2I_OUT_DIR, f"image_image_{image_id}.png")
    t2i_img = Image.open(t2i_path).convert("RGB") if os.path.exists(t2i_path) else None

    # Get caption
    caption = ""
    if os.path.exists(CORRECTED_CAPTIONS_PATH):
        with open(CORRECTED_CAPTIONS_PATH, "r") as f:
            cc = json.load(f)
        # The key mapping: keys are 0-indexed, images start at 96
        for k, v in cc.items():
            if int(k) + 96 == image_id:
                caption = v.get("corrected", v.get("original", ""))
                break

    if not caption and os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, "r") as f:
            caps = json.load(f)
        caption = caps.get(str(image_id), "")

    details = (
        f"**Image ID:** {image_id}\n\n"
        f"**Caption used for generation:**\n> {caption}\n\n"
        f"**Model:** Stable Diffusion XL (1024×1024 → 256×256)\n\n"
        f"**Paper Reference (Section III-D):** The text decoder at the receiver "
        f"converts the corrected semantic description back into a visual image "
        f"using a pre-trained text-to-image generation model."
    )

    return orig_img, t2i_img, details


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: DJSCC Baseline Visualization
# ═══════════════════════════════════════════════════════════════════════════
def show_djscc_results(image_id):
    """Show original vs DJSCC reconstructed image."""
    if image_id is None:
        return None, None, "⚠️ Select an image ID"

    image_id = int(image_id)

    # Original from testdata/resizedata
    orig_path = os.path.join(RESIZE_DIR, f"image_{image_id}.png")
    orig_img = Image.open(orig_path).convert("RGB") if os.path.exists(orig_path) else None

    # DJSCC output
    djscc_path = os.path.join(DJSCC_OUT_DIR, f"image{image_id}.png")
    djscc_img = Image.open(djscc_path).convert("RGB") if os.path.exists(djscc_path) else None

    # Calculate PSNR if both images exist
    psnr_val = "N/A"
    if orig_img and djscc_img and os.path.exists(orig_path) and os.path.exists(djscc_path):
        try:
            psnr_val = f"{calculate_psnr(orig_path, djscc_path):.2f} dB"
        except Exception:
            psnr_val = "Error"

    details = (
        f"**Image ID:** {image_id}\n\n"
        f"**PSNR:** {psnr_val}\n\n"
        f"**Model:** Deep JSCC Autoencoder (CNN Encoder → Channel → CNN Decoder)\n\n"
        f"**Architecture:**\n"
        f"- Encoder: 5-layer CNN (3→16→32→64→128→256 channels)\n"
        f"- Decoder: 5-layer Transposed CNN (256→128→64→32→16→3 channels)\n\n"
        f"**Paper Reference:** DeepJSCC serves as the baseline for conventional "
        f"image transmission without semantic understanding."
    )

    return orig_img, djscc_img, details


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: Evaluation — Compare ULSC vs DJSCC
# ═══════════════════════════════════════════════════════════════════════════
def run_comparison(image_id):
    """Compare original, ULSC reconstruction, and DJSCC reconstruction."""
    if image_id is None:
        return None, None, None, "⚠️ Select an image ID"

    image_id = int(image_id)

    # Original
    orig_path = os.path.join(RESIZE_DIR, f"image_{image_id}.png")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(OURDATASET_DIR, f"{image_id}.png")
    orig_img = Image.open(orig_path).convert("RGB") if os.path.exists(orig_path) else None

    # ULSC (Text-to-Image reconstruction)
    ulsc_path = os.path.join(T2I_OUT_DIR, f"image_image_{image_id}.png")
    ulsc_img = Image.open(ulsc_path).convert("RGB") if os.path.exists(ulsc_path) else None

    # DJSCC
    djscc_path = os.path.join(DJSCC_OUT_DIR, f"image{image_id}.png")
    djscc_img = Image.open(djscc_path).convert("RGB") if os.path.exists(djscc_path) else None

    # Metrics
    metrics_lines = [f"## Comparison for Image {image_id}\n"]

    if os.path.exists(orig_path):
        if ulsc_img and os.path.exists(ulsc_path):
            try:
                ulsc_psnr = calculate_psnr(orig_path, ulsc_path)
                metrics_lines.append(f"**ULSC PSNR:** {ulsc_psnr:.2f} dB")
            except Exception as e:
                metrics_lines.append(f"**ULSC PSNR:** Error ({e})")
        else:
            metrics_lines.append("**ULSC:** No reconstruction found")

        if djscc_img and os.path.exists(djscc_path):
            try:
                djscc_psnr = calculate_psnr(orig_path, djscc_path)
                metrics_lines.append(f"**DJSCC PSNR:** {djscc_psnr:.2f} dB")
            except Exception as e:
                metrics_lines.append(f"**DJSCC PSNR:** Error ({e})")
        else:
            metrics_lines.append("**DJSCC:** No reconstruction found")

    # Caption info
    caption = ""
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, "r") as f:
            caps = json.load(f)
        caption = caps.get(str(image_id), "")

    if caption:
        metrics_lines.append(f"\n**BLIP Caption:** \"{caption}\"")

    metrics_lines.append(
        "\n---\n**Key Insight:** ULSC transmits semantic descriptions (text) instead of "
        "raw pixels, achieving much lower bandwidth while preserving semantic meaning. "
        "DJSCC transmits compressed pixel data through a learned channel codec."
    )

    return orig_img, ulsc_img, djscc_img, "\n\n".join(metrics_lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Show existing result graphs
# ═══════════════════════════════════════════════════════════════════════════
def get_result_graphs():
    """Load existing result graphs from the project."""
    graphs = {}
    graph_files = [
        ("Semantic Similarity vs Erase Rate (4 Curves)", "fig7_4curve_similarity.png"),
        ("Semantic Similarity vs Erase Rate", "fig7_similarity_vs_erase_rate.png"),
        ("PSNR vs Erase Rate (4 Curves)", "fig8_4curve_psnr.png"),
        ("PSNR vs Erase Rate", "fig8_psnr_vs_erase_rate.png"),
        ("Option A - Similarity", "optionA_fig7_similarity.png"),
        ("Option A - Attacker PSNR", "optionA_fig8_attacker_psnr.png"),
    ]
    for label, fname in graph_files:
        fpath = os.path.join(PROJECT_ROOT, fname)
        if os.path.exists(fpath):
            graphs[label] = fpath
    return graphs


# ═══════════════════════════════════════════════════════════════════════════
#  BUILD THE GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════
def build_app():
    available_ids = get_available_image_ids()
    # Filter for the demo range (96-100)
    demo_ids = [str(i) for i in available_ids if 96 <= i <= 100]
    if not demo_ids:
        demo_ids = [str(i) for i in range(96, 101)]

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="ULSC Pipeline Demo — Semantic Communications",
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0f172a",
            body_background_fill_dark="#0f172a",
            block_background_fill="#1e293b",
            block_background_fill_dark="#1e293b",
            block_border_color="rgba(129, 140, 248, 0.15)",
            block_border_color_dark="rgba(129, 140, 248, 0.15)",
            block_label_text_color="#a5b4fc",
            block_label_text_color_dark="#a5b4fc",
            block_title_text_color="#e0e7ff",
            block_title_text_color_dark="#e0e7ff",
            body_text_color="#cbd5e1",
            body_text_color_dark="#cbd5e1",
            button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            input_background_fill="#1e293b",
            input_background_fill_dark="#1e293b",
            input_border_color="rgba(129, 140, 248, 0.3)",
            input_border_color_dark="rgba(129, 140, 248, 0.3)",
            shadow_drop="0 4px 20px rgba(0,0,0,0.3)",
            shadow_drop_lg="0 8px 32px rgba(0,0,0,0.4)",
        )
    ) as demo:

        # ── Title Banner ────────────────────────────────────────────────
        gr.HTML("""
        <div id="title-banner">
            <h1>🛰️ ULSC Pipeline Demo</h1>
            <p>Semantic Importance-Aware Communications with Semantic Correction Using Large Language Models</p>
            <div style="margin-top: 16px; display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                <span class="flow-step">📷 Input Image</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">🔤 BLIP Captioning</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">📡 Channel (Frame Loss)</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">🧠 BERT Correction</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">🖼️ Image Reconstruction</span>
            </div>
        </div>
        """)

        with gr.Tabs():
            # ════════════════════════════════════════════════════════════
            # TAB 1: Image → Text (BLIP)
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🔤 Step 1: Image → Text", id="tab_i2t"):
                gr.HTML('<div class="step-badge">Step 1 — Transmitter Side</div>')
                gr.Markdown(
                    "### BLIP Image Captioning (Image Caption Neural Network)\n"
                    "The Image Encoder (Vision Transformer) converts the input image into "
                    "feature vectors, which are then decoded into a natural language description "
                    "by the Text Decoder."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        i2t_image_id = gr.Dropdown(
                            choices=demo_ids,
                            label="Select Image ID",
                            value=demo_ids[0] if demo_ids else None,
                            interactive=True,
                        )
                        i2t_run_btn = gr.Button("🚀 Generate Caption", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        i2t_image_out = gr.Image(label="Input Image", type="pil", height=300)

                    with gr.Column(scale=2):
                        i2t_details = gr.Markdown(label="Results")
                        i2t_caption_out = gr.Textbox(label="Generated Caption (copy for Step 2)", interactive=True)

                i2t_run_btn.click(
                    fn=run_image_to_text,
                    inputs=[i2t_image_id],
                    outputs=[i2t_image_out, i2t_details, i2t_caption_out],
                )

                gr.HTML("<hr style='border-color: rgba(129,140,248,0.2); margin: 24px 0;'>")
                gr.Markdown("### Batch Mode — Caption Multiple Images")
                with gr.Row():
                    batch_start = gr.Number(label="Start ID", value=96, precision=0)
                    batch_end = gr.Number(label="End ID", value=100, precision=0)
                    batch_i2t_btn = gr.Button("🚀 Run Batch Captioning", variant="primary")
                batch_i2t_output = gr.Markdown(label="Batch Results")
                batch_i2t_btn.click(
                    fn=run_batch_i2t,
                    inputs=[batch_start, batch_end],
                    outputs=[batch_i2t_output],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 2: BERT Semantic Correction
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🧠 Step 2: BERT Correction", id="tab_bert"):
                gr.HTML('<div class="step-badge">Step 2 — Receiver Side (Semantic Correction)</div>')
                gr.Markdown(
                    "### BERT Masked Language Model — Semantic Corrector\n"
                    "Simulates channel frame loss by masking pairs of words, then uses BERT to predict "
                    "the missing words. This demonstrates the LLM-based semantic correction capability "
                    "described in the paper."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        bert_caption_input = gr.Textbox(
                            label="Caption to Correct",
                            placeholder="Paste a caption from Step 1, or type your own...",
                            lines=3,
                        )
                        bert_run_btn = gr.Button("🧠 Run BERT Correction", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        bert_output = gr.Markdown(label="BERT Correction Results")

                bert_run_btn.click(
                    fn=run_bert_correction,
                    inputs=[bert_caption_input],
                    outputs=[bert_output],
                )

                gr.HTML("<hr style='border-color: rgba(129,140,248,0.2); margin: 24px 0;'>")
                gr.Markdown("### Batch Mode — Correct All Captions from captions.json")
                batch_bert_btn = gr.Button("🧠 Run Batch BERT Correction", variant="primary")
                batch_bert_output = gr.Markdown(label="Batch BERT Results")
                batch_bert_btn.click(
                    fn=run_batch_bert,
                    inputs=[],
                    outputs=[batch_bert_output],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 3: Text → Image (Stable Diffusion)
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🖼️ Step 3: Text → Image", id="tab_t2i"):
                gr.HTML('<div class="step-badge">Step 3 — Receiver Side (Image Reconstruction)</div>')
                gr.Markdown(
                    "### Stable Diffusion — Text-to-Image Reconstruction\n"
                    "The corrected caption from BERT is used to reconstruct the image at the receiver "
                    "using a pre-trained Stable Diffusion model. This shows the **pre-computed results** "
                    "from the pipeline (SDXL requires GPU)."
                )

                with gr.Row():
                    t2i_image_id = gr.Dropdown(
                        choices=demo_ids,
                        label="Select Image ID",
                        value=demo_ids[0] if demo_ids else None,
                        interactive=True,
                    )
                    t2i_show_btn = gr.Button("🔍 Show Results", variant="primary", size="lg")

                with gr.Row():
                    t2i_orig = gr.Image(label="Original Image (Transmitter)", type="pil", height=300)
                    t2i_recon = gr.Image(label="SD Reconstructed Image (Receiver)", type="pil", height=300)

                t2i_details = gr.Markdown(label="Details")

                t2i_show_btn.click(
                    fn=show_t2i_results,
                    inputs=[t2i_image_id],
                    outputs=[t2i_orig, t2i_recon, t2i_details],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 4: DJSCC Baseline
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("📡 DJSCC Baseline", id="tab_djscc"):
                gr.HTML('<div class="step-badge">Baseline — Deep JSCC</div>')
                gr.Markdown(
                    "### Deep Joint Source-Channel Coding (DJSCC)\n"
                    "The conventional baseline that transmits compressed image data directly "
                    "through the channel. Uses a CNN-based autoencoder for joint source-channel coding."
                )

                with gr.Row():
                    djscc_image_id = gr.Dropdown(
                        choices=demo_ids,
                        label="Select Image ID",
                        value=demo_ids[0] if demo_ids else None,
                        interactive=True,
                    )
                    djscc_show_btn = gr.Button("🔍 Show Results", variant="primary", size="lg")

                with gr.Row():
                    djscc_orig = gr.Image(label="Original Image", type="pil", height=300)
                    djscc_recon = gr.Image(label="DJSCC Reconstructed", type="pil", height=300)

                djscc_details = gr.Markdown(label="Details")

                djscc_show_btn.click(
                    fn=show_djscc_results,
                    inputs=[djscc_image_id],
                    outputs=[djscc_orig, djscc_recon, djscc_details],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 5: Comparison & Evaluation
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("📊 Comparison & Metrics", id="tab_compare"):
                gr.HTML('<div class="step-badge">Evaluation</div>')
                gr.Markdown(
                    "### Side-by-Side Comparison: ULSC vs DJSCC\n"
                    "Compare the original image with reconstructions from both the proposed ULSC "
                    "framework and the DJSCC baseline."
                )

                with gr.Row():
                    cmp_image_id = gr.Dropdown(
                        choices=demo_ids,
                        label="Select Image ID",
                        value=demo_ids[0] if demo_ids else None,
                        interactive=True,
                    )
                    cmp_run_btn = gr.Button("📊 Compare", variant="primary", size="lg")

                with gr.Row():
                    cmp_orig = gr.Image(label="Original", type="pil", height=280)
                    cmp_ulsc = gr.Image(label="ULSC (Ours)", type="pil", height=280)
                    cmp_djscc = gr.Image(label="DJSCC Baseline", type="pil", height=280)

                cmp_metrics = gr.Markdown(label="Metrics & Analysis")

                cmp_run_btn.click(
                    fn=run_comparison,
                    inputs=[cmp_image_id],
                    outputs=[cmp_orig, cmp_ulsc, cmp_djscc, cmp_metrics],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 6: Result Graphs
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("📈 Result Graphs", id="tab_graphs"):
                gr.HTML('<div class="step-badge">Published Results</div>')
                gr.Markdown("### Performance Evaluation Graphs\nThese are the graphs generated from the full experiment runs.")

                graphs = get_result_graphs()
                if graphs:
                    for label, fpath in graphs.items():
                        gr.Markdown(f"#### {label}")
                        gr.Image(value=fpath, label=label, height=400, interactive=False)
                else:
                    gr.Markdown("⚠️ No result graphs found in the project root.")

            # ════════════════════════════════════════════════════════════
            # TAB 7: System Architecture
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🏗️ Architecture", id="tab_arch"):
                gr.HTML('<div class="step-badge">System Overview</div>')
                gr.Markdown("""
### ULSC Framework Architecture

The **Unified Language-based Semantic Communication (ULSC)** framework converts visual data
to natural language for transmission, enabling:
- **Ultra-low bandwidth** (text is orders of magnitude smaller than images)
- **Semantic-level understanding** of transmitted content
- **LLM-based error correction** at the receiver

---

### Pipeline Stages

| Stage | Component | Model | Location |
|:------|:----------|:------|:---------|
| **1. Image → Text** | Image Caption Neural Network (ICNN) | BLIP (ViT + MED) | Transmitter |
| **2. Semantic Importance** | BERT Masked LM | bert-base-uncased | Both sides |
| **3. Channel Coding** | Frame-based transmission | Custom | Channel |
| **4. Semantic Correction** | LLM Corrector | BERT MLM | Receiver |
| **5. Text → Image** | Image Generation Neural Network (IGNN) | Stable Diffusion XL | Receiver |

---

### Key Equations from the Paper

- **Semantic Importance (Eq. 3):**  `I(wi) = -log₂ P(wi | w₁,...,wi₋₁,wi₊₁,...,wn)`
- **Frame Loss Probability (Eq. 4):**  `P_loss = 1 - P_correct`
- **Semantic Similarity (Eq. 18):**  `Sim(S, Ŝ) = cos(C_Φ(S), C_Φ(Ŝ))` using CLIP ViT-B/32
- **PSNR (Eq. 17):**  `PSNR = 10 · log₁₀(MAX² / MSE)`

---

### Models Used

| Model | Purpose | Parameters |
|:------|:--------|:-----------|
| **BLIP** (blip_caption, base_coco) | Image captioning | ViT-B encoder + MED decoder |
| **BERT** (bert-base-uncased) | Masked word prediction / semantic correction | 110M params |
| **Stable Diffusion XL** | Text-to-image reconstruction | ~3.5B params |
| **CLIP** (ViT-B/32) | Semantic similarity evaluation | 151M params |
| **DeepJSCC** | Baseline CNN autoencoder | Custom 5-layer encoder/decoder |
                """)

        # ── Footer ──────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #64748b; font-size: 0.8rem; margin-top: 24px;">
            ULSC Pipeline Demo • Semantic Importance-Aware Communications • Built with Gradio
        </div>
        """)

    return demo


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
