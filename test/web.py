import importlib.util
from pathlib import Path
import re
import threading
import time
import traceback
import uuid

import streamlit as st
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    TextIteratorStreamer,
)

# =========================
# 1. Path configuration
# =========================
APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parents[2]

MODEL_DIR = PROJECT_ROOT / "model" / "GenNA"
TOKENIZER_DIR = MODEL_DIR

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Required path not found: {MODEL_DIR}")

# =========================
# 2. Page setup
# =========================
st.set_page_config(page_title="GenNA", layout="wide")
st.title("🧬 A Demo for GenNA")

# =========================
# 3. Global resources (without st.cache_resource)
# =========================
_GLOBAL_MODELS = {}
_GLOBAL_MODELS_LOCK = threading.Lock()

_GENERATION_LOCK = threading.Lock()

_QUEUE_LOCK = threading.Lock()
_QUEUE = []
_ACTIVE_REQUEST_ID = None


# =========================
# 4. Utility functions
# =========================
def format_sequence(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def clean_chunk(chunk: str) -> str:
    chunk = chunk.replace("Ġ", "")
    chunk = re.sub(r"\s+", "", chunk)
    return chunk


def get_available_devices():
    devices = []
    device_labels = {}

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            device_name = f"cuda:{idx}"
            gpu_name = torch.cuda.get_device_name(idx)
            devices.append(device_name)
            device_labels[device_name] = f"{device_name} ({gpu_name})"

    devices.append("cpu")
    device_labels["cpu"] = "cpu"
    return devices, device_labels


def load_model_and_tokenizer_uncached(device_name: str):
    device = torch.device(device_name)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)

    model_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
    }

    if device.type == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        **model_kwargs,
    ).to(device)

    model.eval()
    return model, tokenizer, device


def get_or_create_model(device_name: str):
    with _GLOBAL_MODELS_LOCK:
        if device_name not in _GLOBAL_MODELS:
            model, tokenizer, device = load_model_and_tokenizer_uncached(device_name)
            _GLOBAL_MODELS[device_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
            }

        bundle = _GLOBAL_MODELS[device_name]
        return bundle["model"], bundle["tokenizer"], bundle["device"]


def build_inputs(tokenizer, prompt_text: str, device):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=torch.long) for k, v in inputs.items()}
    return inputs


def ensure_session_ids():
    if "client_id" not in st.session_state:
        st.session_state.client_id = str(uuid.uuid4())
    if "request_id" not in st.session_state:
        st.session_state.request_id = None
    if "queued" not in st.session_state:
        st.session_state.queued = False
    if "generating" not in st.session_state:
        st.session_state.generating = False
    if "last_output" not in st.session_state:
        st.session_state.last_output = ""
    if "model_ready_device" not in st.session_state:
        st.session_state.model_ready_device = None
    if "model_load_error" not in st.session_state:
        st.session_state.model_load_error = None
    if "model_loading_done" not in st.session_state:
        st.session_state.model_loading_done = False


def enqueue_request(request_id: str):
    with _QUEUE_LOCK:
        if request_id not in _QUEUE:
            _QUEUE.append(request_id)


def remove_request(request_id: str):
    global _ACTIVE_REQUEST_ID
    with _QUEUE_LOCK:
        if request_id in _QUEUE:
            _QUEUE.remove(request_id)
        if _ACTIVE_REQUEST_ID == request_id:
            _ACTIVE_REQUEST_ID = None


def get_queue_snapshot(request_id: str):
    with _QUEUE_LOCK:
        queue_copy = list(_QUEUE)
        active_id = _ACTIVE_REQUEST_ID

    if request_id in queue_copy:
        pos = queue_copy.index(request_id) + 1
        waiting_ahead = pos - 1
    else:
        pos = None
        waiting_ahead = None

    return {
        "position": pos,
        "waiting_ahead": waiting_ahead,
        "active_request_id": active_id,
        "queue_length": len(queue_copy),
    }


def wait_until_my_turn(request_id: str, status_placeholder):
    global _ACTIVE_REQUEST_ID

    while True:
        with _QUEUE_LOCK:
            is_head = len(_QUEUE) > 0 and _QUEUE[0] == request_id
            active_busy = _ACTIVE_REQUEST_ID is not None
            queue_len = len(_QUEUE)
            my_pos = _QUEUE.index(request_id) + 1 if request_id in _QUEUE else None

        if is_head and not active_busy:
            with _QUEUE_LOCK:
                is_head = len(_QUEUE) > 0 and _QUEUE[0] == request_id
                active_busy = _ACTIVE_REQUEST_ID is not None
                if is_head and not active_busy:
                    _ACTIVE_REQUEST_ID = request_id
                    return

        if my_pos is not None:
            status_placeholder.warning(
                f"In queue: there are currently {queue_len} request(s) in total. "
                f"Your queue position is #{my_pos}, with {my_pos - 1} request(s) ahead of you."
            )
        else:
            status_placeholder.warning("Request state error: the current request was not found in the queue.")

        time.sleep(0.5)


def generate_worker(model, gen_kwargs, error_holder):
    try:
        with torch.inference_mode():
            with _GENERATION_LOCK:
                model.generate(**gen_kwargs)
    except Exception as e:
        error_holder["error"] = str(e)
        error_holder["traceback"] = traceback.format_exc()


# =========================
# 5. Initialize session
# =========================
ensure_session_ids()

# =========================
# 6. Sidebar parameters
# =========================
available_devices, device_labels = get_available_devices()
default_device_index = 0

with st.sidebar:
    st.header("Generation Parameters")

    line_width = st.slider(
        "Line Width (Display)",
        min_value=40,
        max_value=120,
        value=80,
        step=10,
        help="Number of bases displayed per line.",
    )

    st.divider()

    max_new_tokens = st.number_input(
        "Max New Tokens", min_value=1, max_value=8192, value=1000
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    top_p = st.slider("Top-P", 0.0, 1.0, 0.8)
    top_k = st.number_input("Top-K", min_value=0, value=0)
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 3.0, 1.3)
    no_repeat_ngram_size = st.number_input(
        "No Repeat n-gram Size", min_value=2, value=5
    )

    device_opt = st.selectbox(
        "Device",
        available_devices,
        index=default_device_index,
        format_func=lambda x: device_labels.get(x, x),
    )

# =========================
# 7. Auto-load / auto-switch model
# =========================
st.subheader("Model")

if (
    st.session_state.model_ready_device != device_opt
    or not st.session_state.model_loading_done
):
    try:
        with st.spinner(f"Loading model on {device_labels.get(device_opt, device_opt)}..."):
            model, tokenizer, device = get_or_create_model(device_opt)
        st.session_state.model_ready_device = device_opt
        st.session_state.model_load_error = None
        st.session_state.model_loading_done = True
    except Exception:
        st.session_state.model_ready_device = None
        st.session_state.model_load_error = traceback.format_exc()
        st.session_state.model_loading_done = False

if st.session_state.model_ready_device is not None:
    ready_device = device_labels.get(
        st.session_state.model_ready_device,
        st.session_state.model_ready_device,
    )
    st.success(f"Model ready on {ready_device}")
else:
    st.error("Model failed to load.")
    if st.session_state.model_load_error:
        st.code(st.session_state.model_load_error, language="python")

# =========================
# 8. Input area
# =========================
default_prompt = "RNA, Homo sapiens, histone H2A<seq>"
prompt_text = st.text_area("Input Prompt", value=default_prompt, height=100)

# =========================
# 9. Queue status panel
# =========================
st.subheader("Queue Status")
queue_status_placeholder = st.empty()

if st.session_state.request_id is not None:
    snap = get_queue_snapshot(st.session_state.request_id)
    if snap["position"] is not None:
        if snap["active_request_id"] == st.session_state.request_id:
            queue_status_placeholder.success(
                f"Current status: generating. Current total queue length: {snap['queue_length']}."
            )
        else:
            queue_status_placeholder.info(
                f"Current status: waiting in queue. Your queue position is #{snap['position']}, "
                f"with {snap['waiting_ahead']} request(s) ahead of you."
            )
    else:
        queue_status_placeholder.info("There is currently no queued request for this session.")
else:
    with _QUEUE_LOCK:
        qlen = len(_QUEUE)
        active_id = _ACTIVE_REQUEST_ID
    if active_id is not None:
        queue_status_placeholder.info(
            f"A task is currently being generated. Total queue length: {qlen}."
        )
    else:
        queue_status_placeholder.info("The queue is currently empty.")

# =========================
# 10. Generate button
# =========================
generate_clicked = st.button("Generate Sequence", type="primary")

if generate_clicked:
    if st.session_state.model_ready_device is None:
        st.error("Model is not ready.")
        st.stop()

    if st.session_state.request_id is not None:
        snap = get_queue_snapshot(st.session_state.request_id)
        if (
            snap["position"] is not None
            or snap["active_request_id"] == st.session_state.request_id
        ):
            st.warning("You already have a pending generation request.")
            st.stop()

    try:
        model, tokenizer, device = get_or_create_model(st.session_state.model_ready_device)
    except Exception as e:
        st.error(f"Model acquisition failed: {e}")
        st.stop()

    request_id = f"{st.session_state.client_id}-{uuid.uuid4().hex[:8]}"
    st.session_state.request_id = request_id
    st.session_state.queued = True
    st.session_state.generating = False
    st.session_state.last_output = ""

    enqueue_request(request_id)

    output_placeholder = st.empty()
    status_placeholder = st.empty()

    try:
        wait_until_my_turn(request_id, status_placeholder)
        st.session_state.queued = False
        st.session_state.generating = True

        snap = get_queue_snapshot(request_id)
        current_position = snap["position"] if snap["position"] is not None else 1
        status_placeholder.info(
            f"Generation started: your request position is #{current_position} (now running)."
        )

        inputs = build_inputs(tokenizer, prompt_text, device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        do_sample = temperature > 0.0

        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            top_p=float(top_p) if do_sample else None,
            temperature=float(temperature) if do_sample else None,
            top_k=int(top_k) if do_sample else None,
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=True,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        full_response = ""
        error_holder = {"error": None, "traceback": None}

        worker = threading.Thread(
            target=generate_worker,
            args=(model, gen_kwargs, error_holder),
            daemon=True,
        )
        worker.start()

        status_placeholder.info("Generating...")

        for chunk in streamer:
            chunk = clean_chunk(chunk)
            full_response += chunk
            st.session_state.last_output = full_response

            formatted_response = format_sequence(full_response, width=line_width)
            output_placeholder.code(
                formatted_response,
                language="text",
                line_numbers=True,
            )

        worker.join()

        if error_holder["error"] is not None:
            st.error(f"Runtime Error: {error_holder['error']}")
            st.code(error_holder["traceback"], language="python")
        else:
            st.success("Generation Complete!")

    except Exception as e:
        st.error(f"Runtime Error: {e}")
        st.code(traceback.format_exc(), language="python")

    finally:
        st.session_state.queued = False
        st.session_state.generating = False
        remove_request(request_id)
        st.session_state.request_id = None

# =========================
# 11. Display the most recent output
# =========================
if st.session_state.last_output:
    st.subheader("Last Output")
    st.code(
        format_sequence(st.session_state.last_output, width=line_width),
        language="text",
        line_numbers=True,
    )