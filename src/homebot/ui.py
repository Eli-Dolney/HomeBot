import os
from typing import List, Dict, Tuple, Generator, Optional, Iterable

import gradio as gr
import speech_recognition as sr

# Reuse backend logic/constants from jarvis.py
from homebot import jarvis  # type: ignore

PROFILES = {
    "default": jarvis.SYSTEM_PROMPT,
    "coding": jarvis.CODING_ASSISTANT_SYSTEM,
    "story": jarvis.STORYTELLER_SYSTEM,
}


def build_messages(history: List[Tuple[str, str]], user_text: str, profile_key: str) -> List[Dict[str, str]]:
    system = PROFILES.get(profile_key, jarvis.SYSTEM_PROMPT)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": user_text})
    return messages


_recognizer = sr.Recognizer()
_local_stt = jarvis.LocalSTT(model_size=os.environ.get("UI_WHISPER_MODEL", "small"), compute_type=os.environ.get("UI_STT_COMPUTE", "int8_float16"))


def _transcribe_file(audio_path: Optional[str]) -> str:
    if not audio_path:
        return ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio = _recognizer.record(source)
        return _local_stt.transcribe_audio(audio, use_vad=False)
    except Exception:
        return ""


def respond_once(message: str, profile: str, speak: bool) -> str:
    system = PROFILES.get(profile, jarvis.SYSTEM_PROMPT)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": (message or "").strip()},
    ]
    reply = jarvis.chat_ollama(messages)
    if speak and reply:
        jarvis.TextToSpeech().say(reply)
    return reply


def _stream_chat(history: List[Tuple[str, str]], message: str, profile: str, speak: bool) -> Iterable[str]:
    messages = build_messages(history, (message or "").strip(), profile)
    full: List[str] = [""]
    tts = jarvis.TextToSpeech() if speak else None
    for chunk in jarvis.chat_ollama_stream(messages):
        full[0] += chunk
        yield full[0]
        if tts is not None:
            # speak sentence by sentence
            if "." in full[0] or "\n" in full[0]:
                parts = full[0].split(".")
                # speak all complete sentences except the last (may be partial)
                for sent in parts[:-1]:
                    s = sent.strip()
                    if s:
                        if tts.is_speaking:
                            tts.stop()
                        tts.say(s)
                full[0] = parts[-1]
    # speak any trailing partial
    if tts is not None:
        tail = full[0].strip()
        if tail:
            if tts.is_speaking:
                tts.stop()
            tts.say(tail)
    yield full[0]


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="🧠 Jarvis (Local Chat)") as demo:
        with gr.Tab("Chat"):
            chat = gr.ChatInterface(
                fn=_stream_chat,
                additional_inputs=[
                    gr.Dropdown(["default", "coding", "story"], value="default", label="Profile"),
                    gr.Checkbox(value=True, label="🔊 Speak replies (local)")
                ],
                title="🧠 Jarvis (Local Chat)",
                description=f"Streaming with `{jarvis.MODEL_NAME}` via `{jarvis.OLLAMA_HOST}`.",
                retry_btn=None,
                undo_btn=None,
                clear_btn="Clear",
            )
        with gr.Tab("Voice (Local STT)"):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or upload audio")
            profile = gr.Dropdown(["default", "coding", "story"], value="default", label="Profile")
            speak = gr.Checkbox(value=True, label="🔊 Speak replies (local)")
            out = gr.Textbox(label="Assistant")

            def voice_pipeline(audio_path: Optional[str], profile: str, speak: bool) -> str:
                text = _transcribe_file(audio_path)
                if not text:
                    return "(No speech detected)"
                msgs = [{"role": "system", "content": PROFILES.get(profile, jarvis.SYSTEM_PROMPT)}, {"role": "user", "content": text}]
                reply = jarvis.chat_ollama(msgs)
                if speak and reply:
                    jarvis.TextToSpeech().say(reply)
                return reply

            btn = gr.Button("Transcribe & Reply")
            btn.click(fn=voice_pipeline, inputs=[audio, profile, speak], outputs=out)
    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(share=True)
