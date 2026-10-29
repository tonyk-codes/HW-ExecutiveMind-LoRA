---
title: Executive Mindset AI
sdk: gradio
app_file: app.py
python_version: "3.10"
pinned: false
---

# Executive Mindset AI

An interactive demonstration of a Qwen language model with a published LoRA
adapter for executive-management guidance. The app combines local GPU
inference, runtime adapter verification, optional fal.ai text-to-speech, and a
browser-based 3D avatar.

**Model:** [tony-dev1/HW-ExecutiveMind-LoRA-Qwen3.6-27B](https://huggingface.co/tony-dev1/HW-ExecutiveMind-LoRA-Qwen3.6-27B)

**Base model:** [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)

## What this demonstrates

- Loading a PEFT LoRA adapter on top of a large causal language model.
- Checking adapter metadata before allocating the base model.
- Verifying that LoRA modules and parameters were actually injected.
- Supporting BF16 and 4-bit loading modes for different GPU budgets.
- Streaming status updates through a Gradio workflow.
- Optional speech generation and avatar lip-sync as separate media layers.
- A configuration design that works locally and on Hugging Face Spaces.

## Hardware and cost

The 27B base model is not a lightweight laptop model. Use an NVIDIA GPU with
substantial VRAM; an A100-class Hugging Face Space is the practical reference
configuration for BF16. `MODEL_LOAD_MODE=4bit` can reduce memory requirements,
but it does not make CPU inference practical. Audio requests use the fal.ai API
and may incur provider charges.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
python app.py
```

The app opens on `http://localhost:7860`. `config.json` is ignored by Git, so
it is the convenient local place for settings and secrets. Environment
variables always take precedence, which is the recommended approach for CI and
hosted deployments.

### Configuration

`config.example.json` documents every supported value:

| Key | Purpose |
| --- | --- |
| `base_model` | Base Transformers model; defaults to `Qwen/Qwen3.6-27B`. |
| `lora_adapter` | Hugging Face repo ID or downloaded local adapter directory. |
| `fal_key` | fal.ai API key for speech generation. Optional when audio is disabled. |
| `hf_token` | Hugging Face token when model access requires authentication. |
| `tts_model` | fal.ai TTS model identifier. |
| `model_load_mode` | `bf16` or `4bit`. |
| `merge_lora` | Merge adapter weights after loading in BF16 mode. |
| `max_queue_size` | Maximum queued Gradio jobs. |

For a public repository, never commit a real key. `config.json` is ignored,
but production deployments should use secret managers or Hugging Face Space
Secrets instead:

```bash
export FAL_KEY="your-fal-key"
export HF_TOKEN="your-hugging-face-token"
python app.py
```

The app runs without `FAL_KEY`; it returns text and explains that audio was
skipped. `HF_TOKEN` is only needed when the selected model or adapter is
private or gated.

## Adapter download

The adapter is intentionally **not bundled in this Git repository**. Its
`adapter_model.safetensors` file is large, model hosting has its own licensing
and revision lifecycle, and a public code repository should remain cloneable.
Download it reproducibly when you need a local copy:

```bash
python scripts/download_adapter.py
```

This writes to the ignored `models/executive-mindset-lora/` directory. To use
that copy, set this in `config.json`:

```json
{
	"lora_adapter": "models/executive-mindset-lora"
}
```

You can select another compatible adapter without changing code:

```bash
python scripts/download_adapter.py \
	--repo-id owner/compatible-adapter \
	--output-dir models/compatible-adapter
```

## Hugging Face Spaces

1. Create a Gradio Space and upload this repository.
2. Select an A100-class hardware tier for BF16 inference.
3. Add `FAL_KEY` under Space Settings -> Secrets if audio is enabled.
4. Add `HF_TOKEN` only if the selected model or adapter requires it.
5. Let the Space download the adapter from the default public repository at startup.

Spaces provides `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT`; the app detects
that environment and avoids using `share=True` there.

## Runtime flow

```text
question -> chat template -> Qwen base model + PEFT adapter -> answer
																												|-> fal.ai TTS -> audio
																												`-> TalkingHead avatar
```

At the first request, the app reads `adapter_config.json`, compares the
adapter base model with `BASE_MODEL`, loads the model, and verifies that LoRA
modules were injected. The diagnostics panel exposes those checks so a
deployment failure is visible instead of silently producing base-model output.

## Limitations and responsible use

This is a style demonstration, not an executive decision system. Outputs can
be incorrect, culturally narrow, overconfident, or unsuitable for sensitive
business decisions. Do not send confidential company information to a hosted
model or third-party TTS provider. Review the licenses and usage terms of the
Qwen base model, the adapter, fal.ai, and the avatar assets before commercial
deployment.

The linked Hugging Face model card is currently minimal, so training data,
evaluation metrics, and adapter provenance should be documented there before
the model is presented as production-ready.

## License

This repository is distributed under the license in [LICENSE](LICENSE). That
license does not replace the terms of the upstream base model, adapter, or
third-party services.
