import base64
import html
import inspect
import logging
import mimetypes
import os
import re
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import fal_client
import gradio as gr
import requests
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("huawei_demo")
logger.setLevel(logging.INFO)


def _banner(msg: str) -> None:
    """Always-visible log line: bypasses any logging-handler interception in Spaces."""
    line = f"[LoRA] {msg}"
    print(line, flush=True)
    logger.info(msg)


LORA_STATUS: dict[str, str] = {"state": "Not loaded", "detail": ""}


os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

CONFIG = load_config()
BASE_MODEL = CONFIG.base_model
LORA_ADAPTER = CONFIG.lora_adapter
FAL_KEY = CONFIG.fal_key
HF_TOKEN = CONFIG.hf_token or None
TTS_MODEL = CONFIG.tts_model
MODEL_LOAD_MODE = CONFIG.model_load_mode
MERGE_LORA = CONFIG.merge_lora
MAX_QUEUE_SIZE = CONFIG.max_queue_size
TALKINGHEAD_MODULE_URL = "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@1.7/modules/talkinghead.mjs"
AVATAR_LABEL = "Little Jo Avatar"
AVATAR_URL = "https://raw.githubusercontent.com/met4citizen/TalkingHead/main/avatars/mpfb.glb"
AVATAR_BODY = "F"
AVATAR_MOOD = "happy"

SYSTEM_PROMPT = (
    "你是一位兼具企业战略、组织管理、风险意识、"
    "人才培养与长期主义思维的高层管理者。"
    "请用简体中文直接给出最终答复，内容务实、有战略高度，"
    "同时保持常识、克制与可执行性。涉及国际关系、产业政策、"
    "贸易冲突等议题时，要同时呈现机会、代价、约束与不确定性，"
    "避免宣传口号、绝对化判断、空泛排比与重复句式，适度排比可保留。"
    "核心是还原高层老板的谈话口吻、用词习惯和管理心智："
    "先判断问题背后的管理含义，再给出方向性提醒和取舍。"
    "面对新员工、迟到、沟通、协作等日常管理问题时，"
    "不要写成制度 SOP、培训手册或条款清单，"
    "要像老板在现场带人，讲原则、讲分寸、讲期望，"
    "语言自然、有判断力、有温度，但不啰嗦。"
    "输出 2 到 3 个紧凑短段落，每段只讲一个重点，"
    "默认控制在 200 到 350 个汉字，除非用户明确要求展开。"
    "严禁输出任何思考过程、推理草稿或列点分析草案，"
    "不要使用 <think> 标签，也不要以 'Here is a thinking process'、"
    "'让我先分析' 之类的话开头，直接给出最终答复。"
)

if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def launch_app(app: gr.Blocks) -> None:
    # Per Hugging Face Spaces guide, prefer plain launch(); Spaces sets
    # GRADIO_SERVER_NAME/PORT via env vars. Only pass keywords the installed
    # Gradio version supports, and only set share off-Spaces.
    on_spaces = bool(os.getenv("SPACE_ID"))
    launch_options: dict[str, object] = {
        "show_error": True,
        "show_api": False,
        "share": (not on_spaces) and os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes", "on"},
    }
    if not on_spaces:
        launch_options["server_name"] = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        launch_options["server_port"] = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    supported_options = inspect.signature(app.launch).parameters
    app.launch(**{key: value for key, value in launch_options.items() if key in supported_options})


@lru_cache(maxsize=1)
def load_llm():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected, so Qwen 27B cannot be loaded. Please select Nvidia A100 large in Hugging Face Spaces.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)

    # Sanity-check the LoRA adapter config BEFORE loading the heavy base model,
    # so a misconfigured adapter fails fast with a clear error.
    try:
        peft_config = PeftConfig.from_pretrained(LORA_ADAPTER, token=HF_TOKEN)
        _banner(
            f"adapter config OK | base={getattr(peft_config, 'base_model_name_or_path', '?')} "
            f"r={getattr(peft_config, 'r', '?')} alpha={getattr(peft_config, 'lora_alpha', '?')} "
            f"target_modules={getattr(peft_config, 'target_modules', '?')}"
        )
        adapter_base = getattr(peft_config, "base_model_name_or_path", None)
        if adapter_base and adapter_base != BASE_MODEL:
            _banner(
                f"WARNING: adapter trained on {adapter_base} but BASE_MODEL is {BASE_MODEL}"
            )
    except Exception as cfg_err:
        LORA_STATUS["state"] = "Load failed"
        LORA_STATUS["detail"] = f"Could not read adapter_config.json: {cfg_err}"
        _banner(f"ERROR: failed to load adapter config from {LORA_ADAPTER}: {cfg_err}")
        raise

    if MODEL_LOAD_MODE == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            low_cpu_mem_usage=True,
        )
        _banner(f"attaching adapter {LORA_ADAPTER} (4-bit base, no merge)")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER, token=HF_TOKEN, is_trainable=False)
        _verify_lora_attached(model, merged=False)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        _banner(f"attaching adapter {LORA_ADAPTER} (BF16 base)")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER, token=HF_TOKEN, is_trainable=False)
        _verify_lora_attached(model, merged=False)
        if MERGE_LORA:
            _banner("merging LoRA weights into base model (merge_and_unload)...")
            model = model.merge_and_unload()
            LORA_STATUS["state"] = "Loaded and merged"
            LORA_STATUS["detail"] = f"{LORA_ADAPTER} (merge_and_unload completed)"
            _banner("merge complete")

    model.eval()
    return tokenizer, model


def _verify_lora_attached(model, merged: bool) -> None:
    """Confirm the PEFT adapter actually attached LoRA layers to the base model."""
    lora_module_count = 0
    lora_param_count = 0
    sample_names: list[str] = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if "Lora" in cls and cls != "LoraConfig":
            lora_module_count += 1
            if len(sample_names) < 5:
                sample_names.append(f"{name} ({cls})")
    for name, param in model.named_parameters():
        if "lora_" in name.lower():
            lora_param_count += param.numel()

    if lora_module_count == 0 or lora_param_count == 0:
        LORA_STATUS["state"] = "Load anomaly"
        LORA_STATUS["detail"] = (
            f"PEFT wrapping succeeded, but no LoRA layers were injected (modules={lora_module_count}, params={lora_param_count})."
            " The adapter target_modules likely do not match the base architecture."
        )
        _banner(
            f"ERROR: verification FAILED — modules={lora_module_count}, params={lora_param_count}. "
            f"adapter target_modules likely do not match base architecture."
        )
        raise RuntimeError(
            "LoRA adapter attached no layers/params to the base model. "
            "Check that adapter target_modules match the Qwen architecture."
        )

    LORA_STATUS["state"] = "Loaded" if not merged else "Loaded and merged"
    LORA_STATUS["detail"] = (
        f"{LORA_ADAPTER}: {lora_module_count} LoRA modules, {lora_param_count:,} LoRA parameters"
    )
    _banner(
        f"verification OK | LoRA modules={lora_module_count} params={lora_param_count:,} | "
        f"examples: {', '.join(sample_names[:3])}"
    )


def build_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tokenizer, _ = load_llm()
    # Qwen3-family chat templates accept enable_thinking=False to suppress
    # the <think>...</think> chain-of-thought block. Older tokenizers ignore
    # the kwarg, so we fall back gracefully.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TRAILING_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> chain-of-thought block from model output.

    Also removes a dangling unclosed <think>... at the tail (e.g. when the
    model was truncated mid-thought) and any leading 'Here's a thinking
    process:' style preamble that some Qwen3 finetunes emit as plain text.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_TRAILING_RE.sub("", cleaned)
    # Some finetunes leak the CoT as plain markdown instead of <think> tags.
    # Drop any leading "thinking process" preamble up to the first blank line
    # that introduces the actual answer (heuristic, only triggers if present).
    preamble_re = re.compile(
        r"^\s*(here'?s?\s+(a|the)\s+thinking\s+process[:：].*?\n\s*\n)",
        re.IGNORECASE | re.DOTALL,
    )
    cleaned = preamble_re.sub("", cleaned, count=1)
    return cleaned.strip()


def generate_qwen_answer(question: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    tokenizer, model = load_llm()
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not answer:
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return _strip_thinking(answer)


def _download_remote_file(url: str, default_suffix: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or default_suffix
    file_path = Path(tempfile.gettempdir()) / f"huawei_demo_{uuid.uuid4().hex}{suffix}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    file_path.write_bytes(response.content)
    return str(file_path)


def _file_to_data_uri(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(Path(file_path).read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def generate_tts(text: str) -> tuple[str, str]:
    result = fal_client.subscribe(
        TTS_MODEL,
        arguments={
            "text": text,
            "voice_setting": {
                "voice_id": "Wise_Woman",
                "speed": 1.1,
                "pitch": 0,
                "vol": 1.0,
            },
            "output_format": "url",
        },
    )
    audio_url = result["audio"]["url"]
    local_audio_path = _download_remote_file(audio_url, ".mp3")
    return local_audio_path, audio_url


def build_avatar_html(audio_path: str | None = None, enabled: bool = True) -> str:
    if not enabled:
        return """
<div class="avatar-disabled">
    Avatar is disabled
</div>
"""

    audio_data_uri = _file_to_data_uri(audio_path) if audio_path else ""
    has_audio = "true" if audio_data_uri and enabled else "false"
    play_label = "Play avatar voice" if audio_data_uri and enabled else "Waiting for audio"
    disabled_attr = "" if audio_data_uri and enabled else "disabled"

    iframe_doc = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #eef5ff; background: linear-gradient(150deg, #0c2236 0%, #184a73 58%, #2a6d9a 100%); }}
    #stage {{ position: absolute; inset: 0; }}
    #loading {{ position: absolute; inset: 0; display: grid; place-items: center; padding: 24px; text-align: center; color: rgba(238, 245, 255, 0.8); background: radial-gradient(circle at 50% 18%, rgba(255,255,255,0.14), rgba(255,255,255,0)); }}
    #toolbar {{ position: absolute; left: 16px; right: 16px; bottom: 16px; display: flex; align-items: center; gap: 10px; padding: 10px; border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; background: rgba(7, 27, 49, 0.74); backdrop-filter: blur(12px); }}
    button {{ height: 40px; min-width: 136px; border: 0; border-radius: 10px; color: #102034; background: #f2cf7a; font-weight: 700; cursor: pointer; }}
    button:disabled {{ opacity: 0.56; cursor: default; }}
    #state {{ min-width: 0; flex: 1; color: rgba(238, 245, 255, 0.82); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    audio {{ display: none; }}
  </style>
  <script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js/+esm",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/",
      "talkinghead": "{TALKINGHEAD_MODULE_URL}"
    }}
  }}
  </script>
</head>
<body>
  <div id="stage"></div>
    <div id="loading">Loading 3D avatar...</div>
  <div id="toolbar">
    <button id="play" {disabled_attr}>{html.escape(play_label)}</button>
        <div id="state">{html.escape(AVATAR_LABEL)}</div>
  </div>
    <audio id="speech" preload="auto" autoplay playsinline src="{audio_data_uri}"></audio>
  <script type="module">
    import {{ TalkingHead }} from "talkinghead";
    const stage = document.getElementById("stage"), loading = document.getElementById("loading"), playButton = document.getElementById("play"), state = document.getElementById("state"), audio = document.getElementById("speech");
    const hasAudio = {has_audio};
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
        let head, audioCtx, analyser, source, animationFrame, autoplayed = false, sourceConnected = false;
    function setState(message) {{ state.textContent = message; }}
    function setBlendshape(name, value) {{ if (head?.mtAvatar?.[name]) Object.assign(head.mtAvatar[name], {{ newvalue: value, needsUpdate: true }}); }}
        function withTimeout(promise, ms, message) {{
            return Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms))]);
        }}
    function driveMouth() {{
      if (!analyser) return;
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (const sample of data) {{ const centered = (sample - 128) / 128; sum += centered * centered; }}
      const open = Math.min(0.85, Math.max(0, (Math.sqrt(sum / data.length) - 0.015) * 7.5));
      setBlendshape("viseme_aa", open); setBlendshape("viseme_O", open * 0.35); setBlendshape("mouthOpen", open * 0.75);
      animationFrame = requestAnimationFrame(driveMouth);
    }}
        function driveMouthFallback() {{
            if (audio.paused || audio.ended) return;
            const open = 0.12 + Math.random() * 0.5;
            setBlendshape("viseme_aa", open); setBlendshape("viseme_O", open * 0.3); setBlendshape("mouthOpen", open * 0.7);
            animationFrame = requestAnimationFrame(driveMouthFallback);
        }}
    async function prepareAudio() {{
      if (!hasAudio) return;
            if (isIOS) return;
            if (!audioCtx) {{ audioCtx = new AudioContext(); analyser = audioCtx.createAnalyser(); analyser.fftSize = 256; }}
            if (!sourceConnected) {{ source = audioCtx.createMediaElementSource(audio); source.connect(analyser); analyser.connect(audioCtx.destination); sourceConnected = true; }}
            if (audioCtx.state === "suspended") await withTimeout(audioCtx.resume(), 3000, "Audio context startup timed out");
    }}
    async function startPlayback() {{
            if (!hasAudio) return;
      try {{
                setState("Preparing voice playback...");
                                await withTimeout(prepareAudio(), 4000, "Voice playback preparation timed out");
        audio.currentTime = 0;
                                await withTimeout(audio.play(), 4000, "The browser could not start playback");
      }} catch (error) {{
        console.error(error);
                                setState("Click play again, or use the audio control below.");
        playButton.disabled = false;
      }}
    }}
    async function tryAutoplay() {{
      if (!hasAudio || autoplayed) return;
                        if (isIOS) {{ setState("Avatar is ready. Click play to hear the voice."); return; }}
      autoplayed = true;
      await startPlayback();
    }}
    async function initAvatar() {{
      head = new TalkingHead(stage, {{ lipsyncModules: [], cameraView: "upper" }});
                        await head.showAvatar({{ url: "{AVATAR_URL}", body: "{AVATAR_BODY}", avatarMood: "{AVATAR_MOOD}", lipsyncLang: "en" }}, (event) => {{ if (event.lengthComputable) loading.textContent = `Loading 3D avatar... ${{Math.round(event.loaded / event.total * 100)}}%`; }});
      loading.style.display = "none";
                        setState(hasAudio ? (isIOS ? "Avatar is ready. Click play to hear the voice." : "Avatar is ready. Auto-playing voice...") : "Avatar is ready. Waiting for generated audio.");
                        if (hasAudio) audio.load();
      tryAutoplay();
    }}
    playButton.addEventListener("click", startPlayback);
        audio.addEventListener("loadeddata", () => {{ if (head) tryAutoplay(); }}, {{ once: true }});
    audio.addEventListener("canplaythrough", () => {{ if (head) tryAutoplay(); }}, {{ once: true }});
                audio.addEventListener("play", () => {{ playButton.disabled = true; setState("Playing audio and driving lip sync"); cancelAnimationFrame(animationFrame); isIOS ? driveMouthFallback() : driveMouth(); }});
        audio.addEventListener("ended", () => {{ cancelAnimationFrame(animationFrame); setBlendshape("viseme_aa", 0); setBlendshape("viseme_O", 0); setBlendshape("mouthOpen", 0); playButton.disabled = false; setState("Audio playback finished"); }});
        initAvatar().catch((error) => {{ console.error(error); loading.textContent = `Avatar failed to load: ${{error.message || error}}`; setState("Please check the network or avatar asset."); }});
  </script>
</body>
</html>
"""
    escaped_doc = html.escape(iframe_doc, quote=True)
    return f"""
<div class="avatar-frame-wrap">
    <iframe class="avatar-frame" title="TalkingHead 3D avatar" allow="autoplay" srcdoc="{escaped_doc}"></iframe>
</div>
"""


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    return bool(value)


def build_audio_banner(enabled: bool = True) -> str:
    if not enabled:
        return """
<div class="media-disabled">
    Audio is disabled
</div>
"""
    return ""


def format_status(stage: str | None = None, progress: int = 0, started_at: float | None = None) -> str:
    safe_progress = max(0, min(100, int(progress)))
    elapsed = time.time() - started_at if started_at else 0.0
    stage_line = ""
    if stage:
        timer_attr = f" data-started-at='{started_at:.3f}'" if started_at else ""
        stage_line = (
            f"<div>- Current status: {html.escape(stage)}</div>"
            "<div class='status-progress-wrap'>"
            f"<div class='status-progress-fill' style='width: {safe_progress}%;'></div>"
            "</div>"
            f"<div class='status-meta'>Progress: {safe_progress}% · <span class='status-elapsed'{timer_attr}>Elapsed: {elapsed:.1f} s</span></div>"
        )
    gpu = "Yes" if torch.cuda.is_available() else "No"
    lora_detail = f"（{LORA_STATUS['detail']}）" if LORA_STATUS["detail"] else ""
    return (
        "<div class='status-content'>"
        + stage_line
        + "<div class='status-lines'>"
        f"<div>- GPU available: {html.escape(gpu)}</div>"
        f"<div>- Base model: {html.escape(BASE_MODEL)}</div>"
        f"<div>- LoRA adapter: {html.escape(LORA_ADAPTER)}</div>"
        f"<div>- LoRA status: {html.escape(LORA_STATUS['state'] + lora_detail)}</div>"
        "</div></div>"
    )


def update_audio_visibility(enable_tts: bool):
    enable_tts = _as_bool(enable_tts)
    return build_audio_banner(enabled=enable_tts), gr.update(value=None, visible=enable_tts)


def update_avatar_visibility(enable_avatar: bool):
    enable_avatar = _as_bool(enable_avatar)
    if enable_avatar:
        return (
            gr.update(value=True),
            build_audio_banner(enabled=True),
            gr.update(value=None, visible=True),
            build_avatar_html(enabled=True),
        )
    return gr.update(), gr.update(), gr.update(), build_avatar_html(enabled=False)


def run_pipeline(
    question: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    enable_tts: bool,
    enable_avatar: bool,
):
    enable_tts = _as_bool(enable_tts)
    enable_avatar = _as_bool(enable_avatar)
    if enable_avatar:
        enable_tts = True
    if not question or not question.strip():
        yield (
            "Please enter a question before clicking Generate.",
            build_audio_banner(enabled=enable_tts),
            gr.update(value=None, visible=enable_tts),
            build_avatar_html(enabled=enable_avatar),
            format_status("Waiting for a question", progress=0),
        )
        return
    started_at = time.time()
    try:
        yield (
            gr.update(),
            build_audio_banner(enabled=enable_tts),
            gr.update(value=None, visible=enable_tts),
            build_avatar_html(enabled=enable_avatar),
            format_status("Preparing Qwen and LoRA", progress=12, started_at=started_at),
        )
        load_llm()
        yield (
            gr.update(),
            build_audio_banner(enabled=enable_tts),
            gr.update(value=None, visible=enable_tts),
            build_avatar_html(enabled=enable_avatar),
            format_status("Generating management guidance", progress=52, started_at=started_at),
        )
        answer = generate_qwen_answer(question.strip(), temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        audio_path = None
        if enable_tts:
            if not FAL_KEY:
                answer += "\n\n> Note: FAL_KEY was not detected, so text-to-speech was skipped. Add FAL_KEY in Hugging Face Space Secrets."
            else:
                yield (
                    gr.update(),
                    build_audio_banner(enabled=True),
                    gr.update(value=None, visible=True),
                    build_avatar_html(enabled=enable_avatar),
                    format_status("Generating audio", progress=78, started_at=started_at),
                )
                audio_path, _ = generate_tts(answer[:300])
        if enable_avatar and not audio_path and enable_tts:
            answer += "\n\n> Note: Avatar lip sync requires generated audio input. Please check whether audio generation completed successfully."
        avatar_html = build_avatar_html(audio_path, enabled=enable_avatar)
        yield answer, build_audio_banner(enabled=enable_tts), gr.update(value=audio_path, visible=enable_tts), avatar_html, format_status("Generation complete", progress=100, started_at=started_at)
    except Exception as err:
        error_text = f"Generation failed: {err}"
        yield error_text, build_audio_banner(enabled=enable_tts), gr.update(value=None, visible=enable_tts), build_avatar_html(enabled=enable_avatar), format_status("Generation failed", progress=100, started_at=started_at)


CUSTOM_CSS = """
:root {
    --page: #f4f7fb;
    --surface: #ffffff;
    --surface-soft: #f8fbff;
    --ink: #182739;
    --ink-soft: #607084;
    --line: #d8e2ee;
    --gold: #b9893f;
    --deep: #25547f;
    --deep-2: #3d78a9;
    --shadow: 0 14px 34px rgba(30, 58, 88, 0.1);
    color-scheme: light;
}
body, .gradio-container {
    font-family: 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'Source Han Sans CN', sans-serif !important;
    color: var(--ink);
    background: linear-gradient(180deg, #fbfdff 0%, var(--page) 100%) !important;
}
.app-shell { max-width: 1180px; margin: 0 auto; padding: 10px 12px 22px; }
.hero {
    position: relative;
    border: 1px solid #cbd8e6;
    border-radius: 16px;
    padding: 24px 28px 24px 30px;
    color: #112b43;
    background: linear-gradient(135deg, #ffffff 0%, #eef6ff 56%, #e6f1fb 100%);
    box-shadow: var(--shadow);
    overflow: hidden;
}
.hero::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: linear-gradient(180deg, var(--gold), var(--deep)); }
.hero-kicker { margin: 0 0 8px; color: var(--deep); font-size: 0.78rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
.hero-title { font-size: clamp(1.85rem, 3.6vw, 2.35rem); margin: 0; font-weight: 780; line-height: 1.16; letter-spacing: 0; }
.hero-subtitle { margin: 10px 0 0; max-width: 720px; color: var(--ink-soft); font-size: 0.98rem; line-height: 1.65; }
.section-head { margin: 0 0 14px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
.section-title { margin: 0; color: var(--ink); font-size: 1rem; font-weight: 760; line-height: 1.3; }
.section-caption { margin: 5px 0 0; color: var(--ink-soft); font-size: 0.84rem; line-height: 1.5; }
.toggle-row { gap: 12px !important; }
.control-toggle {
    min-height: 44px;
    padding: 10px 14px !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    background: var(--surface-soft) !important;
    transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
}
.control-toggle:hover { border-color: color-mix(in srgb, var(--deep) 45%, var(--line)) !important; box-shadow: 0 4px 12px rgba(37, 84, 127, 0.08); }
.control-toggle label { display: flex !important; align-items: center !important; gap: 10px !important; margin: 0 !important; cursor: pointer; }
.control-toggle input[type="checkbox"] {
    -webkit-appearance: checkbox !important;
    -moz-appearance: checkbox !important;
    appearance: checkbox !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 1 !important;
    visibility: visible !important;
    position: static !important;
    pointer-events: auto !important;
    accent-color: var(--deep);
    cursor: pointer;
}
.control-toggle .label-text { color: var(--ink) !important; font-weight: 650 !important; line-height: 1.2 !important; }
.panel {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: color-mix(in srgb, var(--surface) 94%, transparent);
    box-shadow: var(--shadow);
    overflow: hidden;
}
.panel > .block, .panel > .form, .panel > div { border-radius: 12px !important; }
.primary-btn {
    border: 0 !important;
    border-radius: 10px !important;
    font-size: 1.03rem !important;
    font-weight: 750 !important;
    letter-spacing: 0.02em !important;
    height: 48px !important;
    color: #f6faff !important;
    background: linear-gradient(120deg, var(--deep) 0%, var(--deep-2) 100%) !important;
    box-shadow: 0 12px 24px rgba(27, 82, 128, 0.26) !important;
    transition: transform 140ms ease, box-shadow 200ms ease, filter 160ms ease !important;
}
.primary-btn:hover { filter: brightness(1.06); box-shadow: 0 16px 30px rgba(27, 82, 128, 0.32) !important; transform: translateY(-1px); }
.primary-btn:active { transform: translateY(0); filter: brightness(0.98); }
.answer-box {
    border: 1px solid var(--line);
    border-left: 4px solid var(--gold);
    border-radius: 10px;
    padding: 14px 16px;
    line-height: 1.85;
    background: linear-gradient(180deg, var(--surface) 0%, var(--surface-soft) 100%);
}
.log-window {
    border: 1px solid var(--line);
    border-radius: 10px;
    min-height: 252px;
    max-height: 252px;
    overflow: auto;
    padding: 12px;
    background: var(--surface-soft);
    color: var(--ink-soft);
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.6;
}
.status-content { white-space: normal; }
.status-lines { margin-top: 10px; }
.status-progress-wrap {
    height: 10px;
    margin: 8px 0 4px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: color-mix(in srgb, var(--surface-soft) 78%, #000 4%);
    overflow: hidden;
}
.status-progress-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--deep) 0%, var(--deep-2) 70%, var(--gold) 100%);
    transition: width 220ms ease;
}
.status-meta {
    color: var(--ink-soft);
    font-size: 11px;
    line-height: 1.45;
}
.queue-hint { margin-top: 8px; color: var(--ink-soft); font-size: 0.9rem; line-height: 1.6; }
.media-disabled, .media-status {
    min-height: 92px;
    display: grid;
    place-items: center;
    border: 1px dashed var(--line);
    border-radius: 12px;
    color: var(--ink-soft);
    background: var(--surface-soft);
    font-weight: 650;
}
.media-status {
    border-style: solid;
    color: var(--deep);
    background: color-mix(in srgb, var(--surface-soft) 82%, var(--deep) 18%);
}
.avatar-frame-wrap {
    width: 100%;
    min-height: 460px;
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    background: #0c2236;
    box-shadow: var(--shadow);
}
.avatar-frame { display: block; width: 100%; height: 460px; border: 0; }
.avatar-disabled {
    min-height: 180px;
    display: grid;
    place-items: center;
    border: 1px dashed var(--line);
    border-radius: 14px;
    color: var(--ink-soft);
    background: var(--surface-soft);
}
.gradio-container .gr-markdown,
.gradio-container .gr-prose,
.gradio-container .prose,
.gradio-container .gr-box,
.gradio-container .gr-block,
.gradio-container .gr-group { background: transparent !important; }
.gradio-container .gr-markdown,
.gradio-container .gr-prose,
.gradio-container .prose { border: 0 !important; box-shadow: none !important; }
.gradio-container h3 { color: var(--ink) !important; letter-spacing: 0; }
.gradio-container .panel .gr-markdown,
.gradio-container .panel .gr-box,
.gradio-container .panel .gr-group,
.gradio-container .panel .gr-form,
.gradio-container .panel .gr-block { background: transparent !important; border-color: transparent !important; box-shadow: none !important; }
.gradio-container textarea,
.gradio-container input:not([type="checkbox"]):not([type="radio"]),
.gradio-container select {
    background: var(--surface) !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
    transition: border-color 160ms ease, box-shadow 160ms ease;
}
.gradio-container textarea:focus,
.gradio-container input:not([type="checkbox"]):not([type="radio"]):focus,
.gradio-container select:focus {
    border-color: var(--deep) !important;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--deep) 22%, transparent) !important;
    outline: none !important;
}
.panel { transition: box-shadow 220ms ease, transform 220ms ease; }
.panel:hover { box-shadow: 0 18px 40px rgba(30, 58, 88, 0.12); }
footer { display: none !important; }
html.dark,
body.dark,
.dark,
[data-theme="dark"] {
    --page: #0f141a;
    --surface: #171d24;
    --surface-soft: #202832;
    --ink: #f2f5f7;
    --ink-soft: #b7c2cf;
    --line: #35414f;
    --gold: #e2b45f;
    --deep: #48a3a7;
    --deep-2: #6ec0be;
    --shadow: 0 16px 36px rgba(0, 0, 0, 0.34);
    color-scheme: dark;
}
html.dark body,
body.dark,
.dark .gradio-container,
[data-theme="dark"] .gradio-container {
    background: linear-gradient(180deg, #111820 0%, var(--page) 100%) !important;
}
html.dark .hero,
body.dark .hero,
.dark .hero,
[data-theme="dark"] .hero {
    color: var(--ink);
    border-color: #435263;
    background: linear-gradient(135deg, #1b222b 0%, #222b36 54%, #283542 100%);
}
html.dark .hero-kicker,
body.dark .hero-kicker,
.dark .hero-kicker,
[data-theme="dark"] .hero-kicker { color: var(--gold); }
html.dark .hero-subtitle,
body.dark .hero-subtitle,
.dark .hero-subtitle,
[data-theme="dark"] .hero-subtitle { color: var(--ink-soft); }
html.dark .panel,
body.dark .panel,
.dark .panel,
[data-theme="dark"] .panel {
    background: color-mix(in srgb, var(--surface) 96%, #000 4%);
    border-color: var(--line);
}
html.dark .panel:hover,
body.dark .panel:hover,
.dark .panel:hover,
[data-theme="dark"] .panel:hover { box-shadow: 0 18px 40px rgba(0, 0, 0, 0.38); }
html.dark .answer-box,
body.dark .answer-box,
.dark .answer-box,
[data-theme="dark"] .answer-box {
    background: linear-gradient(180deg, #1a212a 0%, #202832 100%);
    color: var(--ink);
}
html.dark .control-toggle,
body.dark .control-toggle,
.dark .control-toggle,
[data-theme="dark"] .control-toggle { background: #1c242e !important; }
html.dark .media-status,
body.dark .media-status,
.dark .media-status,
[data-theme="dark"] .media-status {
    color: #94d8d3;
    background: color-mix(in srgb, var(--surface-soft) 80%, var(--deep) 20%);
}
html.dark .primary-btn,
body.dark .primary-btn,
.dark .primary-btn,
[data-theme="dark"] .primary-btn {
    color: #071214 !important;
    background: linear-gradient(120deg, var(--gold) 0%, #8fd3cf 100%) !important;
    box-shadow: 0 12px 24px rgba(72, 163, 167, 0.24) !important;
}
@media (max-width: 720px) {
    .app-shell { padding: 8px 8px 18px; }
    .hero { padding: 18px 16px; border-radius: 14px; }
    .hero-title { font-size: 2rem; line-height: 1.16; }
    .panel { border-radius: 12px; }
    .avatar-frame-wrap { min-height: 380px; }
    .avatar-frame { height: 380px; }
}
"""


APP_JS = """
() => {
    const url = new URL(window.location.href);
    if (!url.searchParams.has("__theme")) {
        url.searchParams.set("__theme", "light");
        window.location.replace(url.toString());
        return;
    }

    function refreshElapsedTimers() {
        document.querySelectorAll(".status-elapsed[data-started-at]").forEach((element) => {
            const startedAt = Number(element.getAttribute("data-started-at"));
            if (!Number.isFinite(startedAt)) return;
            const elapsed = Math.max(0, Date.now() / 1000 - startedAt);
            element.textContent = `Elapsed: ${elapsed.toFixed(1)} s`;
        });
    }

    refreshElapsedTimers();
    setInterval(refreshElapsedTimers, 500);
    const observer = new MutationObserver(refreshElapsedTimers);
    observer.observe(document.body, { childList: true, subtree: true });
}
"""


def build_ui():
    with gr.Blocks(title="Huawei Executive Mindset LLM", js=APP_JS) as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.HTML(
            """
            <div class='app-shell'>
              <section class='hero'>
                                <p class='hero-kicker'>Executive Mindset AI</p>
                <h1 class='hero-title'>Huawei Executive Mindset LLM</h1>
              </section>
            </div>
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["panel"]):
                    gr.HTML("<div class='section-head'><h2 class='section-title'>Management Question</h2></div>")
                    question = gr.Textbox(lines=6, placeholder="Example: In an uncertain market environment, how should a company maintain strategic focus while balancing growth and cash-flow safety?", label="Enter your question")
                    with gr.Row():
                        temperature = gr.Slider(0.2, 0.7, value=0.5, step=0.05, label="Creativity (Temperature)")
                        top_p = gr.Slider(0.45, 0.85, value=0.65, step=0.05, label="Sampling range (Top-p)")
                        max_new_tokens = gr.Slider(96, 384, value=224, step=16, label="Maximum answer length")
                    with gr.Row(elem_classes=["toggle-row"]):
                        enable_tts = gr.Checkbox(value=True, label="Generate audio", elem_classes=["control-toggle"])
                        enable_avatar = gr.Checkbox(value=True, label="Enable Little Jo avatar", elem_classes=["control-toggle"])
                    run_btn = gr.Button("Generate answer", elem_classes=["primary-btn"])
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["panel"]):
                    gr.HTML("<div class='section-head'><h2 class='section-title'>Runtime Diagnostics</h2><p class='section-caption'>LoRA adapter loading status</p></div>")
                    status_box = gr.HTML(value=format_status(), elem_classes=["log-window"])
        with gr.Group(elem_classes=["panel"]):
            gr.HTML("<div class='section-head'><h2 class='section-title'>Huawei Executive Mindset LLM Answer</h2></div>")
            answer_output = gr.Markdown("<div class='answer-box'>Generation will start after you enter a question.</div>")
        with gr.Group(elem_classes=["panel"]):
            gr.HTML("<div class='section-head'><h2 class='section-title'>Audio</h2></div>")
            audio_banner = gr.HTML(value=build_audio_banner(enabled=True))
            audio_output = gr.Audio(label="Audio", type="filepath")
        with gr.Group(elem_classes=["panel"]):
            gr.HTML("<div class='section-head'><h2 class='section-title'>Little Jo Avatar</h2></div>")
            avatar_output = gr.HTML(value=build_avatar_html())
        run_btn.click(
            fn=run_pipeline,
            inputs=[question, temperature, top_p, max_new_tokens, enable_tts, enable_avatar],
            outputs=[answer_output, audio_banner, audio_output, avatar_output, status_box],
            show_progress="hidden",
        )
        enable_tts.change(fn=update_audio_visibility, inputs=enable_tts, outputs=[audio_banner, audio_output])
        enable_avatar.change(
            fn=update_avatar_visibility,
            inputs=enable_avatar,
            outputs=[enable_tts, audio_banner, audio_output, avatar_output],
        )
    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1, max_size=MAX_QUEUE_SIZE)
    launch_app(app)