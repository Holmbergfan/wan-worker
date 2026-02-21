# wan-worker

A [RunPod](https://www.runpod.io/) serverless worker that runs [WAN 2.1](https://github.com/Wan-Video/Wan2.1) video generation models for **Text-to-Video (T2V)** and **Image-to-Video (I2V)** tasks via the 🤗 [Diffusers](https://github.com/huggingface/diffusers) library.

---

## Supported models

| Model ID | Task | VRAM |
|---|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` *(default T2V)* | Text → Video | ~8 GB |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Text → Video | ~40 GB |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` *(default I2V)* | Image → Video | ~40 GB |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | Image → Video | ~40 GB |

---

## Input schema

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `task` | `string` | No | `"t2v"` | `"t2v"` (text-to-video) or `"i2v"` (image-to-video) |
| `prompt` | `string` | **Yes** | – | Text prompt describing the desired video |
| `negative_prompt` | `string` | No | `null` | Negative prompt to guide generation away from |
| `model_id` | `string` | No | task default | HuggingFace model repo ID |
| `image` | `string` | I2V only | – | Base64-encoded image **or** an `http(s)://` URL |
| `num_frames` | `int` | No | `16` | Number of frames to generate |
| `height` | `int` | No | `480` | Frame height in pixels |
| `width` | `int` | No | `832` | Frame width in pixels |
| `guidance_scale` | `float` | No | `5.0` | Classifier-free guidance scale |
| `num_inference_steps` | `int` | No | `50` | Denoising steps |
| `fps` | `int` | No | `16` | Frames-per-second of the output video |
| `seed` | `int` | No | random | Random seed for reproducibility |

## Output schema

| Field | Type | Description |
|---|---|---|
| `video_base64` | `string` | Base64-encoded `.mp4` video |
| `seed` | `int` | Seed used (only present when `seed` was provided) |
| `error` | `string` | Error message (only present on failure) |

---

## Local testing

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended: 24 GB+ VRAM for 14B models; 8 GB for 1.3B T2V)

```bash
pip install -r requirements.txt
```

### Run with test_input.json

```bash
python handler.py --test_input test_input.json
```

Or pass input directly:

```bash
python handler.py --test_input '{"input": {"task": "t2v", "prompt": "A sunset over the ocean", "num_frames": 16, "seed": 0}}'
```

---

## Docker

### Build

```bash
docker build -t wan-worker .
```

### Run locally

```bash
docker run --gpus all -e RUNPOD_WEBHOOK_GET_JOB="" wan-worker
```

---

## Deploying to RunPod Serverless

1. Push the image to a container registry (e.g. Docker Hub or GitHub Container Registry).
2. In the RunPod console, create a new **Serverless Endpoint** and point it at your image.
3. Attach a **Network Volume** pre-loaded with the WAN model weights if you want to avoid downloading models on every cold start.
4. Set the GPU type (RTX 4090 or A100 recommended) and desired worker count.
5. Send requests to the endpoint:

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "t2v",
      "prompt": "A futuristic city at night, neon lights reflecting on wet streets",
      "num_frames": 24,
      "height": 480,
      "width": 832,
      "seed": 42
    }
  }'
```

The response contains `video_base64`. Decode it to obtain the `.mp4` file:

```python
import base64, json

result = json.loads(response.text)
with open("output.mp4", "wb") as f:
    f.write(base64.b64decode(result["output"]["video_base64"]))
```