import streamlit as st
import requests
import json
from datetime import datetime
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://127.0.0.1:11434"
CHAT_API = f"{OLLAMA_HOST}/api/chat"
TAGS_API = f"{OLLAMA_HOST}/api/tags"      # to list local models

DEFAULT_MODEL = "gemma3:1b"               # change if you have others
HISTORY_TURNS_TO_KEEP = 10                # trims oldest turns to fit context

st.set_page_config(page_title="💬 Abhi's Chatbot", page_icon="🤖")
st.title("🤖 My Chatbot (Ollama + Streamlit)" )

# ── Health check ───────────────────────────────────────────────────────────────
def healthy() -> bool:
    try:
        r = requests.get(OLLAMA_HOST, timeout=3)
        return r.status_code == 200 and "Ollama is running" in r.text
    except Exception:
        return False

if not healthy():
    st.error("⚠️ Ollama server not reachable at 127.0.0.1:11434. Start it with `ollama serve`.")
    st.stop()

# ── Sidebar: controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Settings")

    # Try to list local models; fall back to default if it fails
    models = [DEFAULT_MODEL]
    try:
        tags = requests.get(TAGS_API, timeout=5).json().get("models", [])
        models = [m["name"] for m in tags] or models
    except Exception:
        pass

    MODEL = st.selectbox("Model", models, index=min(models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0, len(models)-1))
    SYSTEM_PROMPT = st.text_area("System Prompt", value="You are a helpful, concise assistant.", height=90)
    TEMP = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    TOP_P = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    NUM_PREDICT = st.slider("Max tokens to generate", 64, 2048, 512, 32)

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with colB:
        st.write("")  # spacer

    st.divider()
    st.caption("💾 Save/Load transcript")
    save_name = st.text_input("Filename (no spaces)", value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    if st.button("Save"):
        Path("chats").mkdir(exist_ok=True)
        path = Path("chats") / save_name
        with path.open("w", encoding="utf-8") as f:
            json.dump(st.session_state.get("messages", []), f, ensure_ascii=False, indent=2)
        st.success(f"Saved to {path}")
    load_files = [str(p) for p in Path("chats").glob("*.json")]
    load_choice = st.selectbox("Load file", ["(choose)"] + load_files)
    if st.button("Load") and load_choice != "(choose)":
        with open(load_choice, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
        st.success(f"Loaded {load_choice}")
        st.rerun()

# ── Session state for messages ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role": "user"/"assistant", "content": "..."}

# ── Helpers ───────────────────────────────────────────────────────────────────
def trimmed_history(history, keep_last=HISTORY_TURNS_TO_KEEP):
    # keep system + last N user/assistant turns
    sys = [{"role": "system", "content": SYSTEM_PROMPT}]
    non_sys = [m for m in history if m["role"] in ("user", "assistant")]
    if len(non_sys) > 2 * keep_last:
        non_sys = non_sys[-2 * keep_last:]
    return sys + non_sys

def stream_chat(messages):
    """Call Ollama /api/chat with streaming."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": float(TEMP),
            "top_p": float(TOP_P),
            "num_predict": int(NUM_PREDICT),
        },
    }
    with requests.post(CHAT_API, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            # Each line is a JSON object prefixed by "data: "
            txt = line.decode("utf-8").removeprefix("data: ").strip()
            try:
                obj = json.loads(txt)
            except json.JSONDecodeError:
                continue
            if "message" in obj and "content" in obj["message"]:
                yield obj["message"]["content"]
            if obj.get("done"):
                break

# ── Render history ────────────────────────────────────────────────────────────
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_msg = st.chat_input("Type your message…")
if user_msg:
    # append user turn
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # build messages for /api/chat (system + trimmed history + new user turn)
    msgs = trimmed_history(st.session_state.messages)

    # assistant bubble + streaming
    with st.chat_message("assistant"):
        box = st.empty()
        reply = ""
        try:
            for chunk in stream_chat(msgs):
                reply += chunk
                box.markdown(reply or "_thinking…_")
        except Exception as e:
            reply = f"⚠️ Error talking to model: `{e}`"
            box.markdown(reply)

    # append assistant turn
    st.session_state.messages.append({"role": "assistant", "content": reply})
