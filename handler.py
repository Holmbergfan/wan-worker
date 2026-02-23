"""RunPod serverless worker for WAN video generation models.

Supports:
- Text-to-Video (T2V) via WanPipeline
- Image-to-Video (I2V) via WanImageToVideoPipeline
"""

import base64
import io
import os
import shutil
import tempfile

from PIL import Image
import requests
import runpod
from huggingface_hub import snapshot_download
import torch
from diffusers import WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image

# ---------------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------------
T2V_MODELS = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
}

I2V_MODELS = {
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
}

DEFAULT_T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_I2V_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Conservative on-disk requirements used for preflight checks before download.
# Values include headroom because snapshot_download may use temporary files.
MODEL_REQUIRED_GB = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": 16,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": 65,
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": 25,
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": 65,
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": 65,
}

# Local /runpod-volume paths written by snapshot_download (via /api/setup).
# Checked before falling back to HuggingFace download.
LOCAL_MODEL_PATHS = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers":    "/runpod-volume/models/wan-t2v-1.3b",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers":     "/runpod-volume/models/wan-t2v-14b",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers":     "/runpod-volume/models/wan-ti2v-5b",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":"/runpod-volume/models/wan-i2v-14b-480p",
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers":"/runpod-volume/models/wan-i2v-14b-720p",
}

# Short aliases accepted from clients (e.g. server.js sends "wan22-ti2v-5b").
MODEL_ALIASES = {
    "wan21-t2v-1.3b":    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan21-t2v-14b":     "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "wan22-ti2v-5b":     "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "wan21-i2v-14b-480p":"Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "wan21-i2v-14b-720p":"Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
}

# ---------------------------------------------------------------------------
# Global pipeline cache – one pipeline loaded at a time.
# ---------------------------------------------------------------------------
_pipeline = None
_pipeline_model_id = None


def _bytes_to_gb(value: int) -> float:
    return value / (1024 ** 3)


def _model_is_ready(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "model_index.json"))


def _download_preflight(hf_repo: str, output_dir: str, required_gb_hint=None):
    output_dir = os.path.abspath(output_dir)
    output_parent = os.path.dirname(output_dir) or output_dir
    os.makedirs(output_parent, exist_ok=True)

    required_gb = required_gb_hint
    if required_gb is not None:
        try:
            required_gb = float(required_gb)
        except (TypeError, ValueError):
            required_gb = None
    if required_gb is None:
        required_gb = MODEL_REQUIRED_GB.get(hf_repo)

    usage = shutil.disk_usage(output_parent)
    free_gb = _bytes_to_gb(usage.free)

    if required_gb is None:
        # Unknown model size: still enforce a small baseline and report free space.
        required_gb = 10

    if free_gb < required_gb:
        return {
            "ok": False,
            "free_gb": round(free_gb, 2),
            "required_gb": required_gb,
            "path": output_parent,
        }

    return {
        "ok": True,
        "free_gb": round(free_gb, 2),
        "required_gb": required_gb,
        "path": output_parent,
    }


def _get_pipeline(model_id: str, task: str):
    """Load (or return cached) pipeline for *model_id*."""
    global _pipeline, _pipeline_model_id

    if _pipeline is not None and _pipeline_model_id == model_id:
        return _pipeline

    # Unload previous pipeline to free VRAM before loading a new one.
    if _pipeline is not None:
        del _pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _pipeline = None
        _pipeline_model_id = None

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prefer locally downloaded copy on /runpod-volume over HuggingFace.
    local_path = LOCAL_MODEL_PATHS.get(model_id)
    local_ready = bool(local_path and _model_is_ready(local_path))
    if local_path and os.path.isdir(local_path) and not local_ready:
        print(f"Warning: local model path exists but appears incomplete: {local_path}")
    load_path = local_path if local_ready else model_id
    print(f"Loading pipeline: {model_id} (task={task}, dtype={dtype}, path={load_path})")

    # I2V tasks always need WanImageToVideoPipeline (accepts 'image' kwarg).
    # T2V tasks use WanPipeline. Both work with TI2V and dedicated models.
    if task == "i2v":
        pipe = WanImageToVideoPipeline.from_pretrained(load_path, torch_dtype=dtype)
    else:
        pipe = WanPipeline.from_pretrained(load_path, torch_dtype=dtype)

    pipe.to(device)

    _pipeline = pipe
    _pipeline_model_id = model_id
    return _pipeline


def _encode_video(path: str) -> str:
    """Return the mp4 at *path* as a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """RunPod serverless handler.

    Expected ``job["input"]`` keys:
        task (str): "t2v" or "i2v". Default: "t2v".
        model_id (str): HuggingFace model repo ID.
                        Defaults to DEFAULT_T2V_MODEL / DEFAULT_I2V_MODEL.
        prompt (str): Text prompt describing the desired video. Required.
        negative_prompt (str): Optional negative prompt.
        image (str): Base64-encoded image or URL. Required for "i2v".
        num_frames (int): Number of frames to generate. Default: 16.
        height (int): Frame height in pixels. Default: 480.
        width (int): Frame width in pixels. Default: 832.
        guidance_scale (float): Classifier-free guidance scale. Default: 5.0.
        num_inference_steps (int): Denoising steps. Default: 50.
        seed (int): Random seed for reproducibility. Optional.
        fps (int): Frames-per-second for the saved video. Default: 16.

    Returns a dict with:
        video_base64 (str): Base64-encoded mp4 video.
        seed (int): Seed that was used.
    """
    job_input = job.get("input", {})

    # ------------------------------------------------------------------
    # List available models — fast directory check, no pipeline loading
    # ------------------------------------------------------------------
    if job_input.get("list_models"):
        available = [
            hf_id
            for hf_id, local_path in LOCAL_MODEL_PATHS.items()
            if _model_is_ready(local_path)
        ]
        disk = {}
        try:
            st = os.statvfs("/runpod-volume")
            total_gb = round(st.f_blocks * st.f_frsize / (1024 ** 3), 1)
            free_gb  = round(st.f_bfree  * st.f_frsize / (1024 ** 3), 1)
            used_gb  = round(total_gb - free_gb, 1)
            disk = {
                "total_gb": total_gb,
                "used_gb":  used_gb,
                "free_gb":  free_gb,
                "used_pct": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
            }
        except Exception as exc:
            disk = {"error": str(exc)}
        return {"available_models": available, "disk": disk}

    # ------------------------------------------------------------------
    # Download mode — triggered by the /api/setup endpoint
    # ------------------------------------------------------------------
    if job_input.get("download_models"):
        model_config = job_input.get("model_config", {})
        hf_repo = model_config.get("hf_repo")
        output_dir = model_config.get("output_dir")
        required_gb = model_config.get("required_gb")
        if not hf_repo or not output_dir:
            return {"error": "download_models requires model_config with hf_repo and output_dir"}

        # Skip redownload if the destination already looks complete.
        if os.path.isdir(output_dir) and _model_is_ready(output_dir):
            return {"status": "already_downloaded", "hf_repo": hf_repo, "output_dir": output_dir}

        preflight = _download_preflight(hf_repo, output_dir, required_gb)
        if not preflight["ok"]:
            return {
                "error": (
                    "Insufficient disk space for model download. "
                    f"Need >= {preflight['required_gb']} GB free on {preflight['path']}, "
                    f"have {preflight['free_gb']} GB."
                ),
                "hint": (
                    "Attach/expand a larger network volume, choose a smaller model "
                    "(e.g. wan21-i2v-5b), or clear old models/caches under /runpod-volume/models "
                    "and /runpod-volume/.cache/huggingface."
                ),
            }

        try:
            print(f"[download] {hf_repo} → {output_dir}")
            snapshot_download(repo_id=hf_repo, local_dir=output_dir)
            print(f"[download] done: {output_dir}")
            return {"status": "downloaded", "hf_repo": hf_repo, "output_dir": output_dir}
        except Exception as exc:
            err = str(exc)
            if "No space left on device" in err:
                post = _download_preflight(hf_repo, output_dir, required_gb)
                return {
                    "error": f"Download failed: {err}",
                    "hint": (
                        "Disk is full during download. Free space on /runpod-volume, then retry. "
                        f"Current free space: {post.get('free_gb')} GB."
                    ),
                }
            return {"error": f"Download failed: {err}"}

    # ------------------------------------------------------------------
    # Parse inputs
    # ------------------------------------------------------------------
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A 'prompt' is required."}

    negative_prompt = job_input.get("negative_prompt", None)

    # Resolve model alias (e.g. "wan22-ti2v-5b" -> full HF repo ID).
    raw_model_id = job_input.get("model_id", "")
    model_id = MODEL_ALIASES.get(raw_model_id.lower(), raw_model_id) if raw_model_id else ""

    # Infer task from model when not explicitly provided.
    task = job_input.get("task", "").lower()
    if task not in ("t2v", "i2v"):
        if model_id in I2V_MODELS:
            task = "i2v"
        elif model_id in T2V_MODELS:
            task = "t2v"
        elif "-i2v-" in model_id.lower():
            task = "i2v"
        else:
            task = "t2v"

    # Fall back to per-task default when no model_id was provided.
    if not model_id:
        model_id = DEFAULT_I2V_MODEL if task == "i2v" else DEFAULT_T2V_MODEL

    # Validate model is known (warn but don't block custom models).
    known_models = T2V_MODELS | I2V_MODELS
    if model_id not in known_models:
        print(f"Warning: model_id '{model_id}' is not in the list of known models.")

    num_frames = int(job_input.get("num_frames", 16))
    height = int(job_input.get("height", 480))
    width = int(job_input.get("width", 832))
    guidance_scale = float(job_input.get("guidance_scale", 5.0))
    num_inference_steps = int(job_input.get("num_inference_steps", 50))
    fps = int(job_input.get("fps", 16))

    seed = job_input.get("seed")
    if seed is not None:
        seed = int(seed)
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)
    else:
        generator = None

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    try:
        pipe = _get_pipeline(model_id, task)
    except Exception as exc:
        return {"error": f"Failed to load model '{model_id}': {exc}"}

    # ------------------------------------------------------------------
    # Build inference kwargs
    # ------------------------------------------------------------------
    inference_kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    if negative_prompt:
        inference_kwargs["negative_prompt"] = negative_prompt

    # For I2V tasks, load the conditioning image.
    if task == "i2v":
        image_input = job_input.get("image")
        if not image_input:
            return {"error": "'image' is required for image-to-video tasks."}
        print(f"[i2v] image type={type(image_input).__name__}, len={len(image_input) if isinstance(image_input, str) else '?'}, starts={repr(image_input[:80]) if isinstance(image_input, str) else '?'}")
        try:
            if isinstance(image_input, str) and image_input.strip().startswith(("http://", "https://")):
                url = image_input.strip()
                resp = requests.get(url, timeout=30)
                print(f"[i2v] fetch status={resp.status_code}, content-type={resp.headers.get('content-type')}, body_len={len(resp.content)}")
                if resp.status_code != 200:
                    return {"error": f"Failed to fetch image: HTTP {resp.status_code} — {resp.text[:200]}"}
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                # Assume base64 (strip data URI prefix if present)
                b64 = image_input if isinstance(image_input, str) else str(image_input)
                b64 = b64.split(",", 1)[-1] if "," in b64[:64] else b64
                image_bytes = base64.b64decode(b64)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inference_kwargs["image"] = image
        except Exception as exc:
            return {"error": f"Failed to load image: {exc}"}

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    try:
        output = pipe(**inference_kwargs)
    except Exception as exc:
        return {"error": f"Inference failed: {exc}"}

    # ------------------------------------------------------------------
    # Export video and encode to base64
    # ------------------------------------------------------------------
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        export_to_video(output.frames[0], tmp_path, fps=fps)
        video_b64 = _encode_video(tmp_path)
    except Exception as exc:
        return {"error": f"Failed to export video: {exc}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    result = {"video_base64": video_b64}
    if seed is not None:
        result["seed"] = seed

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
