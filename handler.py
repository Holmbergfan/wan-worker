"""
WAN 2.1 Image-to-Video RunPod serverless worker.

Accepted input shapes
─────────────────────
Image-to-video generation:
  {
    "image": "<URL or data-URI of the source image>",
    "prompt": "optional motion description",
    "model":  "wan-i2v-5b" | "wan-i2v-14b-480p" | "wan-i2v-14b-720p",  (default: wan-i2v-5b)
    "lora":   "filename.safetensors",  (optional – file must be in <MODEL_PATH>/loras/)
    "width":  832,                     (default: 832)
    "height": 480,                     (default: 480)
    "num_frames":          81,         (default: 81, range 1–121)
    "fps":                 16,         (default: 16)
    "guidance_scale":      7.5,        (default: 7.5)
    "num_inference_steps": 30,         (default: 30)
    "motion_strength":     1.0,        (default: 1.0)
    "seed":                -1          (default: -1 = random)
  }

Model pre-download (runs on a pod with a network volume):
  {
    "download_models": true,
    "model": "5b" | "14b-480p" | "14b-720p"   (default: "5b")
  }

Output
──────
  { "video": "data:video/mp4;base64,<...>" }
  or on error:
  { "error": "<message>" }
"""

import base64
import os
import tempfile

import requests
import runpod
import torch
from PIL import Image
from io import BytesIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models")

# Maps the user-facing model key to the HuggingFace repo ID and the local
# directory name under MODEL_PATH.
MODEL_CONFIG = {
    "wan-i2v-5b": {
        "hf_id": "Wan-AI/Wan2.1-I2V-5B",
        "local": "wan-i2v-5b",
    },
    "wan-i2v-14b-480p": {
        "hf_id": "Wan-AI/Wan2.1-I2V-14B-480P",
        "local": "wan-i2v-14b-480p",
    },
    "wan-i2v-14b-720p": {
        "hf_id": "Wan-AI/Wan2.1-I2V-14B-720P",
        "local": "wan-i2v-14b-720p",
    },
}

# Lazy pipeline cache – avoids reloading for consecutive jobs with the same model.
_pipeline_cache: dict = {}

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _model_source(model_key: str) -> str:
    """Return local path if already downloaded, otherwise HF repo ID."""
    cfg = MODEL_CONFIG[model_key]
    local = os.path.join(MODEL_PATH, cfg["local"])
    return local if os.path.isdir(local) else cfg["hf_id"]


def _load_pipeline(model_key: str):
    """Load (or retrieve from cache) the I2V pipeline for *model_key*."""
    if model_key in _pipeline_cache:
        return _pipeline_cache[model_key]

    from diffusers import WanImageToVideoPipeline

    source = _model_source(model_key)
    print(f"[worker] Loading pipeline from: {source}")

    pipe = WanImageToVideoPipeline.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    _pipeline_cache[model_key] = pipe
    return pipe


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _fetch_image(src: str) -> Image.Image:
    """Load a PIL Image from a URL or a data-URI."""
    if src.startswith("data:"):
        header, b64data = src.split(",", 1)
        image_bytes = base64.b64decode(b64data)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    resp = requests.get(src, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _frames_to_base64(frames: list, fps: int) -> str:
    """Encode a list of PIL Images as an MP4 and return a base64 data-URI."""
    import imageio
    import numpy as np

    np_frames = [np.array(f) for f in frames]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        imageio.mimsave(
            tmp_path,
            np_frames,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        with open(tmp_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return f"data:video/mp4;base64,{b64}"


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------


def _generate_i2v(inp: dict) -> dict:
    """Run image-to-video inference and return base64-encoded MP4."""
    model_key = inp.get("model", "wan-i2v-5b")
    if model_key not in MODEL_CONFIG:
        return {"error": f"Unknown model '{model_key}'. Valid options: {list(MODEL_CONFIG)}"}

    pipe = _load_pipeline(model_key)

    # ---- LoRA ---------------------------------------------------------------
    lora_name = inp.get("lora")
    if lora_name:
        lora_path = os.path.join(MODEL_PATH, "loras", lora_name)
        if os.path.isfile(lora_path):
            print(f"[worker] Loading LoRA: {lora_path}")
            pipe.load_lora_weights(lora_path)
        else:
            print(f"[worker] LoRA file not found, skipping: {lora_path}")

    # ---- Source image -------------------------------------------------------
    image = _fetch_image(inp["image"])
    width  = int(inp.get("width",  832))
    height = int(inp.get("height", 480))
    image  = image.resize((width, height), Image.LANCZOS)

    # ---- Generator ----------------------------------------------------------
    seed = int(inp.get("seed", -1))
    generator = None
    if seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # ---- Inference ----------------------------------------------------------
    output = pipe(
        image=image,
        prompt=inp.get("prompt", ""),
        height=height,
        width=width,
        num_frames=int(inp.get("num_frames", 81)),
        guidance_scale=float(inp.get("guidance_scale", 7.5)),
        num_inference_steps=int(inp.get("num_inference_steps", 30)),
        generator=generator,
    )

    # ---- Unload LoRA so next job starts clean -------------------------------
    if lora_name:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

    frames = output.frames[0]  # list of PIL Images
    fps    = int(inp.get("fps", 16))
    video  = _frames_to_base64(frames, fps)
    return {"video": video}


def _download_models(model_variant: str) -> dict:
    """Download model weights to the network volume via huggingface_hub."""
    from huggingface_hub import snapshot_download

    # Normalise variant aliases (e.g. "5b" → "wan-i2v-5b")
    alias_map = {
        "5b":       "wan-i2v-5b",
        "14b-480p": "wan-i2v-14b-480p",
        "14b-720p": "wan-i2v-14b-720p",
    }
    key = alias_map.get(model_variant, model_variant)
    if key not in MODEL_CONFIG:
        return {"error": f"Unknown model variant '{model_variant}'"}

    cfg   = MODEL_CONFIG[key]
    dest  = os.path.join(MODEL_PATH, cfg["local"])
    os.makedirs(dest, exist_ok=True)

    print(f"[worker] Downloading {cfg['hf_id']} → {dest}")
    snapshot_download(
        repo_id=cfg["hf_id"],
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    return {"status": "downloaded", "model": key, "path": dest}


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


def handler(job: dict) -> dict:
    inp = job.get("input", {})

    try:
        if inp.get("download_models"):
            return _download_models(inp.get("model", "5b"))

        if "image" in inp:
            return _generate_i2v(inp)

        return {"error": "Invalid input. Provide 'image' for I2V or 'download_models: true' for setup."}

    except Exception as exc:  # pylint: disable=broad-except
        import traceback
        return {"error": str(exc), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

runpod.serverless.start({"handler": handler})
