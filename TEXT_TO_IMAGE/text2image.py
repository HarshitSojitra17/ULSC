# """
# text2image.py — Prompt → Image generation step
# Pipeline context:
#     resizedata/image96..100.png
#         → IMAGE_TO_TEXT  (BLIP captioning)
#         → BERT_MASK/corrected_captions.json  (BERT correction)
#         → TEXT_TO_IMAGE/outputdata/image_image_96..100.png   ← this script
#         → similarity.py  (CLIP semantic similarity vs originals)

# Run on: Lightning AI (GPU available)
# Model : runwayml/stable-diffusion-v1-5
#   - Native resolution: 512×512  (SD v1.x family)
#   - Output resized to 256×256 via Lanczos to match resizedata dimensions
#   - DO NOT use SDXL at 256×256 — SDXL needs 1024×1024 minimum to produce
#     coherent images; at 256×256 it outputs abstract noise.
# """

# import os
# import json
# import numpy as np
# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from PIL import Image

# # ── Device & dtype ────────────────────────────────────────────────────────────
# if torch.cuda.is_available():
#     device     = torch.device("cuda")
#     torch_dtype = torch.float16   # float16 on GPU: faster, less VRAM
# else:
#     device     = torch.device("cpu")
#     torch_dtype = torch.float32   # float16 is broken on CPU — always float32

# print("=" * 60)
# print("  TEXT → IMAGE  |  SD v1.5  |  Lightning AI")
# print(f"  Device : {device}  |  dtype : {torch_dtype}")
# print("  Scheduler : DPM++ 2M Karras  |  steps=30  |  cfg=7.5")
# print("  Generate at 512×512  →  resize to 256×256 (Lanczos)")
# print("=" * 60)

# # ── Load SD v1.5 pipeline ─────────────────────────────────────────────────────
# # runwayml/stable-diffusion-v1-5 is the best SD v1.x checkpoint for
# # photorealistic scenes (truck, forest, eagle, flowers, beach).
# # It is designed for 512×512 generation — we resize to 256×256 afterwards.
# # This gives far better quality than generating directly at 256×256.
# print("\nLoading SD v1.5 pipeline...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch_dtype,
#     safety_checker=None,               # disabled: causes false-positive black images
#     requires_safety_checker=False,
# )

# # ── DPM++ 2M Karras — sharper, more photorealistic than default PNDM ─────────
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#     pipe.scheduler.config,
#     use_karras_sigmas=True,
#     algorithm_type="dpmsolver++"
# )

# pipe.to(device)
# pipe.enable_attention_slicing()        # lower peak VRAM usage
# print(f"Pipeline ready  (device={device}).\n")

# # ── Load corrected captions ───────────────────────────────────────────────────
# captions_path = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "..", "BERT_MASK", "corrected_captions.json"
# )
# if not os.path.exists(captions_path):
#     print(f"ERROR: corrected_captions.json not found at:\n  {captions_path}")
#     exit(1)

# with open(captions_path, "r") as f:
#     all_captions = json.load(f)

# # keys are "0","1",... → map to image numbers 96,97,...
# prompts = {k: v["corrected"] for k, v in all_captions.items()}
# print(f"Loaded {len(prompts)} corrected captions:")
# for k, v in prompts.items():
#     print(f"  [{int(k)+96}] {v}")
# print()

# # ── Prompt engineering ────────────────────────────────────────────────────────
# # Quality suffix guides SD toward photorealistic output (vs cartoon/painting).
# QUALITY_SUFFIX = (
#     ", photorealistic, high quality, highly detailed, "
#     "sharp focus, natural lighting, 8k"
# )
# NEGATIVE_PROMPT = (
#     "cartoon, painting, illustration, anime, sketch, drawing, "
#     "low quality, blurry, noisy, distorted, deformed, ugly, "
#     "watermark, text, logo, oversaturated, jpeg artifacts"
# )

# # ── Output directory ──────────────────────────────────────────────────────────
# # similarity.py expects: TEXT_TO_IMAGE/outputdata/image_image_{96..100}.png
# output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputdata")
# os.makedirs(output_dir, exist_ok=True)

# # ── Generation loop ───────────────────────────────────────────────────────────
# # Strategy: generate at SD v1.5 native 512×512, then Lanczos-resize to 256×256.
# # Generating natively at 256×256 produces blurry, incoherent images because
# # the latent space (256÷8 = 32 pixels) is too small to encode scene structure.

# GEN_SIZE    = 512           # SD v1.x native resolution
# TARGET_SIZE = (256, 256)    # final size expected by similarity.py

# total = len(prompts)
# for idx, (key, base_prompt) in enumerate(prompts.items()):
#     image_num = int(key) + 96   # key "0" → image_image_96.png
#     out_path  = os.path.join(output_dir, f"image_image_{image_num}.png")

#     full_prompt = base_prompt + QUALITY_SUFFIX

#     print(f"[{idx+1}/{total}] image_image_{image_num}.png")
#     print(f"  Prompt   : '{base_prompt}'")
#     print(f"  Generate : {GEN_SIZE}×{GEN_SIZE}  →  resize to {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")

#     with torch.no_grad():
#         result = pipe(
#             full_prompt,
#             negative_prompt=NEGATIVE_PROMPT,
#             height=GEN_SIZE,
#             width=GEN_SIZE,
#             num_inference_steps=30,    # DPM++ 2M Karras converges well in 30 steps
#             guidance_scale=7.5,        # standard SD v1.x value; 9+ causes over-saturation
#         )
#     image = result.images[0]           # PIL Image at 512×512

#     # Lanczos is the highest-quality downscaling filter in PIL
#     image = image.resize(TARGET_SIZE, Image.LANCZOS)

#     # Sanity check for all-black output (safety_checker artifacts)
#     arr = np.array(image, dtype=np.float32)
#     if arr.max() < 5:
#         print(f"  ⚠  WARNING: Near-black image (max pixel = {arr.max():.1f})")
#     else:
#         print(f"  ✓  OK  — pixel max={arr.max():.0f}  mean={arr.mean():.1f}")

#     image.save(out_path)
#     print(f"  Saved → {out_path}\n")

# print("=" * 60)
# print(f"  Done. {total} images saved to:")
# print(f"  {output_dir}")
# print("=" * 60)

import os
import json
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image

# ── Device setup (Lightning AI has GPU — use float16 for speed) ───────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.float16    # float16 is fine on GPU — faster & less VRAM
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32    # float16 is broken on CPU — must use float32

print("=" * 60)
print("  Stable Diffusion XL — High Quality Generation")
print("  Scheduler: DPM++ 2M Karras | guidance_scale=7.5 | 30 steps")
print(f"  Device: {device} | dtype: {torch_dtype}")
print("=" * 60)

# ── Load SDXL pipeline ────────────────────────────────────────────────────────
# KEY FIX: SDXL is trained at 1024×1024. Generating at 256×256 directly
# produces garbage (abstract/glitchy images) because the latent space
# collapses at that resolution. Correct approach:
#   1. Generate at SDXL native resolution (1024×1024)
#   2. Resize down to 256×256 using high-quality Lanczos resampling
#
# NOTE: SDXL does NOT have safety_checker / requires_safety_checker params —
# those are SD v1.x only. Passing them to SDXL causes silent errors.
print("\nLoading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant="fp16" if torch_dtype == torch.float16 else None,
)

# ── DPM++ 2M Karras scheduler — sharper than default Euler ───────────────────
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++"
)

pipe.to(device)
pipe.enable_attention_slicing()     # reduces VRAM peak usage during inference
print(f"Pipeline loaded OK (DPM++ 2M Karras, device={device}).\n")

# ── Load corrected captions ───────────────────────────────────────────────────
captions_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BERT_MASK", "corrected_captions.json"
)
if os.path.exists(captions_path):
    with open(captions_path, "r") as f:
        all_captions = json.load(f)
    prompts = {k: v["corrected"] for k, v in all_captions.items()}
    print(f"Loaded {len(prompts)} corrected captions")
else:
    print("ERROR: corrected_captions.json not found!")
    exit(1)

# ── Quality suffix and negative prompt ────────────────────────────────────────
QUALITY_SUFFIX = ", photorealistic, highly detailed, 4k, sharp focus, natural lighting"
NEGATIVE_PROMPT = (
    "cartoon, painting, illustration, anime, sketch, low quality, "
    "blurry, distorted, ugly, watermark, text, oversaturated, "
    "tiling, jpeg artifacts, deformed"
)

# ── Create output directory ───────────────────────────────────────────────────
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputdata")
os.makedirs(output_dir, exist_ok=True)

# ── Generate all images ───────────────────────────────────────────────────────
TARGET_SIZE = (256, 256)   # final saved size required by downstream pipeline
GEN_SIZE    = 1024         # SDXL native resolution — DO NOT lower this

total = len(prompts)
for idx, (key, base_prompt) in enumerate(prompts.items()):
    image_num = int(key) + 96
    out_path = os.path.join(output_dir, f"image_image_{image_num}.png")

    prompt = base_prompt + QUALITY_SUFFIX

    print(f"\n[{idx+1}/{total}] Generating image {image_num}")
    print(f"  Prompt : '{base_prompt}'")
    print(f"  -> Generate at {GEN_SIZE}×{GEN_SIZE}, resize to {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")

    with torch.no_grad():
        result = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=GEN_SIZE,
            width=GEN_SIZE,
            num_inference_steps=30,    # DPM++ 2M Karras converges in 30 steps
            guidance_scale=7.5         # standard SDXL value — 9.5+ causes over-saturation
        )
    image = result.images[0]   # PIL Image at 1024×1024

    # ── Resize from 1024×1024 → 256×256 using Lanczos (best for downscaling) ─
    image = image.resize(TARGET_SIZE, Image.LANCZOS)

    # ── Validate (check for all-black output) ─────────────────────────────────
    img_array = np.array(image, dtype=np.float32)
    if img_array.max() < 5:
        print(f"  WARNING: Near-black image (max={img_array.max():.1f})")
    else:
        print(f"  OK — pixel max={img_array.max():.0f}, mean={img_array.mean():.1f}")

    image.save(out_path)
    print(f"  Saved -> {out_path}")

print(f"\n{'=' * 60}")
print(f"  All {total} images saved to: {output_dir}")
print(f"{'=' * 60}")


# """
# text2image.py — Prompt → Image generation step
# Pipeline context:
#     resizedata/image96..100.png
#         → IMAGE_TO_TEXT  (BLIP captioning)
#         → BERT_MASK/corrected_captions.json  (BERT correction)
#         → TEXT_TO_IMAGE/outputdata/image_image_96..100.png   ← this script
#         → similarity.py  (CLIP semantic similarity vs originals)

# Run on: Lightning AI (GPU available)
# Model : runwayml/stable-diffusion-v1-5
#   - Native resolution: 512×512  (SD v1.x family)
#   - Output resized to 256×256 via Lanczos to match resizedata dimensions
#   - DO NOT use SDXL at 256×256 — SDXL needs 1024×1024 minimum to produce
#     coherent images; at 256×256 it outputs abstract noise.
# """

# import os
# import json
# import numpy as np
# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from PIL import Image

# # ── Device & dtype ────────────────────────────────────────────────────────────
# if torch.cuda.is_available():
#     device     = torch.device("cuda")
#     torch_dtype = torch.float16   # float16 on GPU: faster, less VRAM
# else:
#     device     = torch.device("cpu")
#     torch_dtype = torch.float32   # float16 is broken on CPU — always float32

# print("=" * 60)
# print("  TEXT → IMAGE  |  SD v1.5  |  Lightning AI")
# print(f"  Device : {device}  |  dtype : {torch_dtype}")
# print("  Scheduler : DPM++ 2M Karras  |  steps=30  |  cfg=7.5")
# print("  Generate at 512×512  →  resize to 256×256 (Lanczos)")
# print("=" * 60)

# # ── Load SD v1.5 pipeline ─────────────────────────────────────────────────────
# # runwayml/stable-diffusion-v1-5 is the best SD v1.x checkpoint for
# # photorealistic scenes (truck, forest, eagle, flowers, beach).
# # It is designed for 512×512 generation — we resize to 256×256 afterwards.
# # This gives far better quality than generating directly at 256×256.
# print("\nLoading SD v1.5 pipeline...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch_dtype,
#     safety_checker=None,               # disabled: causes false-positive black images
#     requires_safety_checker=False,
# )

# # ── DPM++ 2M Karras — sharper, more photorealistic than default PNDM ─────────
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#     pipe.scheduler.config,
#     use_karras_sigmas=True,
#     algorithm_type="dpmsolver++"
# )

# pipe.to(device)
# pipe.enable_attention_slicing()        # lower peak VRAM usage
# print(f"Pipeline ready  (device={device}).\n")

# # ── Load corrected captions ───────────────────────────────────────────────────
# captions_path = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "..", "BERT_MASK", "corrected_captions.json"
# )
# if not os.path.exists(captions_path):
#     print(f"ERROR: corrected_captions.json not found at:\n  {captions_path}")
#     exit(1)

# with open(captions_path, "r") as f:
#     all_captions = json.load(f)

# # keys are "0","1",... → map to image numbers 96,97,...
# prompts = {k: v["corrected"] for k, v in all_captions.items()}
# print(f"Loaded {len(prompts)} corrected captions:")
# for k, v in prompts.items():
#     print(f"  [{int(k)+96}] {v}")
# print()

# # ── Prompt engineering ────────────────────────────────────────────────────────
# # Quality suffix guides SD toward photorealistic output (vs cartoon/painting).
# QUALITY_SUFFIX = (
#     ", photorealistic, high quality, highly detailed, "
#     "sharp focus, natural lighting, 8k"
# )
# NEGATIVE_PROMPT = (
#     "cartoon, painting, illustration, anime, sketch, drawing, "
#     "low quality, blurry, noisy, distorted, deformed, ugly, "
#     "watermark, text, logo, oversaturated, jpeg artifacts"
# )

# # ── Output directory ──────────────────────────────────────────────────────────
# # similarity.py expects: TEXT_TO_IMAGE/outputdata/image_image_{96..100}.png
# output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputdata")
# os.makedirs(output_dir, exist_ok=True)

# # ── Generation loop ───────────────────────────────────────────────────────────
# # Strategy: generate at SD v1.5 native 512×512, then Lanczos-resize to 256×256.
# # Generating natively at 256×256 produces blurry, incoherent images because
# # the latent space (256÷8 = 32 pixels) is too small to encode scene structure.

# GEN_SIZE    = 512           # SD v1.x native resolution
# TARGET_SIZE = (256, 256)    # final size expected by similarity.py

# total = len(prompts)
# for idx, (key, base_prompt) in enumerate(prompts.items()):
#     image_num = int(key) + 96   # key "0" → image_image_96.png
#     out_path  = os.path.join(output_dir, f"image_image_{image_num}.png")

#     full_prompt = base_prompt + QUALITY_SUFFIX

#     print(f"[{idx+1}/{total}] image_image_{image_num}.png")
#     print(f"  Prompt   : '{base_prompt}'")
#     print(f"  Generate : {GEN_SIZE}×{GEN_SIZE}  →  resize to {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")

#     with torch.no_grad():
#         result = pipe(
#             full_prompt,
#             negative_prompt=NEGATIVE_PROMPT,
#             height=GEN_SIZE,
#             width=GEN_SIZE,
#             num_inference_steps=30,    # DPM++ 2M Karras converges well in 30 steps
#             guidance_scale=7.5,        # standard SD v1.x value; 9+ causes over-saturation
#         )
#     image = result.images[0]           # PIL Image at 512×512

#     # Lanczos is the highest-quality downscaling filter in PIL
#     image = image.resize(TARGET_SIZE, Image.LANCZOS)

#     # Sanity check for all-black output (safety_checker artifacts)
#     arr = np.array(image, dtype=np.float32)
#     if arr.max() < 5:
#         print(f"  ⚠  WARNING: Near-black image (max pixel = {arr.max():.1f})")
#     else:
#         print(f"  ✓  OK  — pixel max={arr.max():.0f}  mean={arr.mean():.1f}")

#     image.save(out_path)
#     print(f"  Saved → {out_path}\n")

# print("=" * 60)
# print(f"  Done. {total} images saved to:")
# print(f"  {output_dir}")
# print("=" * 60)