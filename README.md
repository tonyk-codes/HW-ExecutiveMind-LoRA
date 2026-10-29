---
title: Executive Mindset AI
sdk: gradio
app_file: app.py
python_version: "3.10"
pinned: false
---

# Executive Mindset AI

This Hugging Face Space runs a Gradio app from `app.py`.

After the Space finishes building, open the normal Space URL to view the app. Do not use `share=True` on Hugging Face Spaces; Spaces serves the Gradio app directly.

Required Space secrets:

- `FAL_KEY`: fal.ai API key for TTS.
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`: optional, needed if the base model or LoRA adapter requires authenticated access.

Recommended hardware: Nvidia A100 large.
