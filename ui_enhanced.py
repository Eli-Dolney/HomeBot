import os
import time
import threading
import queue
from typing import List, Dict, Tuple, Generator, Optional, Iterable
import tempfile
import wave

import gradio as gr
import speech_recognition as sr
import numpy as np

# Reuse backend logic/constants from jarvis.py
import jarvis  # type: ignore

PROFILES = {
    "default": jarvis.SYSTEM_PROMPT,
    "coding": jarvis.CODING_ASSISTANT_SYSTEM,
    "story": jarvis.STORYTELLER_SYSTEM,
}

# Global state for real-time audio processing
_audio_queue = queue.Queue()
_is_listening = False
_recognizer = sr.Recognizer()
_local_stt = jarvis.LocalSTT(
    model_size=os.environ.get("UI_WHISPER_MODEL", "small"), 
    compute_type=os.environ.get("UI_STT_COMPUTE", "auto")
)

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

def _transcribe_file(audio_path: Optional[str]) -> str:
    if not audio_path:
        return ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio = _recognizer.record(source)
        return _local_stt.transcribe_audio(audio, use_vad=True)
    except Exception as exc:
        return f"(Transcription error: {exc})"

def _stream_chat(history: List[Tuple[str, str]], message: str, profile: str, speak: bool) -> Iterable[str]:
    """Enhanced streaming chat with better sentence handling"""
    messages = build_messages(history, (message or "").strip(), profile)
    full_text = ""
    partial_sentence = ""
    tts = jarvis.TextToSpeech() if speak else None
    
    try:
        for chunk in jarvis.chat_ollama_stream(messages):
            full_text += chunk
            partial_sentence += chunk
            
            # Check for sentence boundaries
            while "." in partial_sentence or "\n" in partial_sentence or "!" in partial_sentence or "?" in partial_sentence:
                # Find the first sentence ending
                for sep in [".", "!", "?", "\n"]:
                    if sep in partial_sentence:
                        sentence, partial_sentence = partial_sentence.split(sep, 1)
                        sentence = sentence.strip()
                        if sentence and tts:
                            if tts.is_speaking:
                                tts.stop()
                            tts.say(sentence + sep)
                        break
            
            yield full_text
        
        # Speak any remaining partial sentence
        if tts and partial_sentence.strip():
            if tts.is_speaking:
                tts.stop()
            tts.say(partial_sentence.strip())
            
    except Exception as exc:
        yield f"(Streaming error: {exc})"
    
    yield full_text

def _real_time_audio_processor():
    """Background thread for real-time audio processing"""
    global _is_listening, _audio_queue
    
    while _is_listening:
        try:
            # Get audio data from queue
            audio_data = _audio_queue.get(timeout=0.1)
            if audio_data is None:  # Shutdown signal
                break
                
            # Process audio chunk
            # This would integrate with real-time STT
            # For now, we'll use the existing file-based approach
            
        except queue.Empty:
            continue
        except Exception as exc:
            print(f"Audio processing error: {exc}")

def start_listening():
    """Start real-time listening"""
    global _is_listening
    _is_listening = True
    return "🎤 Listening... (Click Stop to end)"

def stop_listening():
    """Stop real-time listening"""
    global _is_listening
    _is_listening = False
    return "⏹️ Stopped listening"

def create_enhanced_ui() -> gr.Blocks:
    """Create enhanced UI with streaming, real-time audio, and better UX"""
    
    with gr.Blocks(
        title="🧠 Jarvis Enhanced (Mac M4 Optimized)",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #f3e5f5;
            margin-right: 20%;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🧠 Jarvis Enhanced - Mac M4 Optimized
        
        **Privacy-first local AI assistant** with streaming responses, real-time voice input, and Mac-specific tools.
        
        - 🎤 **Real-time voice input** with local STT
        - 🚀 **Streaming responses** for instant feedback  
        - 🛠️ **Mac-specific tools** (Spotlight, Calendar, Notes, etc.)
        - 🔒 **Fully local** - no data leaves your Mac
        - ⚡ **M4 optimized** with MPS acceleration
        """)
        
        with gr.Tabs():
            # Enhanced Chat Tab
            with gr.Tab("💬 Enhanced Chat", id="chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    profile_dropdown = gr.Dropdown(
                        choices=["default", "coding", "story"],
                        value="default",
                        label="🤖 Assistant Profile",
                        scale=1
                    )
                    speak_checkbox = gr.Checkbox(
                        value=True,
                        label="🔊 Speak Replies",
                        scale=1
                    )
                    stream_checkbox = gr.Checkbox(
                        value=True,
                        label="🚀 Stream Responses",
                        scale=1
                    )
                
                # Chat functionality
                def chat_response(message, history, profile, speak, stream):
                    if not message.strip():
                        return history, ""
                    
                    if stream:
                        # Streaming response
                        history.append((message, ""))
                        full_response = ""
                        for chunk in _stream_chat(history[:-1], message, profile, speak):
                            full_response = chunk
                            history[-1] = (message, full_response)
                            yield history, ""
                        return history, ""
                    else:
                        # Non-streaming response
                        messages = build_messages(history, message, profile)
                        response = jarvis.chat_ollama(messages)
                        if speak and response:
                            jarvis.TextToSpeech().say(response)
                        history.append((message, response))
                        return history, ""
                
                # Event handlers
                send_btn.click(
                    chat_response,
                    inputs=[msg_input, chatbot, profile_dropdown, speak_checkbox, stream_checkbox],
                    outputs=[chatbot, msg_input]
                )
                
                msg_input.submit(
                    chat_response,
                    inputs=[msg_input, chatbot, profile_dropdown, speak_checkbox, stream_checkbox],
                    outputs=[chatbot, msg_input]
                )
            
            # Enhanced Voice Tab
            with gr.Tab("🎤 Voice Input", id="voice"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎤 Record or Upload Audio",
                            format="wav"
                        )
                        
                        with gr.Row():
                            transcribe_btn = gr.Button("🎯 Transcribe & Reply", variant="primary")
                            clear_audio_btn = gr.Button("🗑️ Clear")
                    
                    with gr.Column(scale=1):
                        voice_profile = gr.Dropdown(
                            choices=["default", "coding", "story"],
                            value="default",
                            label="🤖 Profile"
                        )
                        voice_speak = gr.Checkbox(
                            value=True,
                            label="🔊 Speak Reply"
                        )
                        voice_stream = gr.Checkbox(
                            value=True,
                            label="🚀 Stream Response"
                        )
                
                transcription_output = gr.Textbox(
                    label="📝 Transcription",
                    lines=2,
                    interactive=False
                )
                
                voice_response = gr.Textbox(
                    label="🤖 Assistant Response",
                    lines=6,
                    interactive=False
                )
                
                def voice_pipeline(audio_path, profile, speak, stream):
                    if not audio_path:
                        return "", ""
                    
                    # Transcribe audio
                    text = _transcribe_file(audio_path)
                    if not text:
                        return "", "(No speech detected or transcription failed)"
                    
                    # Generate response
                    if stream:
                        messages = build_messages([], text, profile)
                        full_response = ""
                        for chunk in jarvis.chat_ollama_stream(messages):
                            full_response = chunk
                            if speak and chunk:
                                # Simple TTS for streaming (could be enhanced)
                                pass
                        if speak and full_response:
                            jarvis.TextToSpeech().say(full_response)
                        return text, full_response
                    else:
                        messages = [{"role": "system", "content": PROFILES.get(profile, jarvis.SYSTEM_PROMPT)}, {"role": "user", "content": text}]
                        response = jarvis.chat_ollama(messages)
                        if speak and response:
                            jarvis.TextToSpeech().say(response)
                        return text, response
                
                transcribe_btn.click(
                    voice_pipeline,
                    inputs=[audio_input, voice_profile, voice_speak, voice_stream],
                    outputs=[transcription_output, voice_response]
                )
                
                clear_audio_btn.click(
                    lambda: (None, "", ""),
                    outputs=[audio_input, transcription_output, voice_response]
                )
            
            # Tools Tab
            with gr.Tab("🛠️ Mac Tools", id="tools"):
                gr.Markdown("""
                ### Available Mac-Specific Tools
                
                You can use these voice commands with Jarvis:
                
                **System Control:**
                - "Open Calculator" - Opens Calculator app
                - "Open Safari" - Opens Safari browser
                - "Lock screen" - Locks your Mac
                - "Set volume to 50" - Adjusts system volume
                
                **Search & Files:**
                - "Search for [query]" - Opens Spotlight search
                - "Show desktop" - Shows desktop
                - "Empty trash" - Empties trash
                
                **Calendar & Notes:**
                - "Open Calendar" - Opens Calendar app
                - "Create calendar event [title]" - Creates new event
                - "Create note [content]" - Creates new note
                
                **Media Control:**
                - "Play music" - Starts Music playback
                - "Pause music" - Pauses Music
                - "Next track" - Skips to next song
                
                **Development:**
                - "Open Terminal" - Opens Terminal
                - "Open Code" - Opens VS Code
                - "Open Xcode" - Opens Xcode
                """)
                
                with gr.Row():
                    test_tool_btn = gr.Button("🧪 Test Tool: Open Calculator")
                    system_info_btn = gr.Button("📊 Show System Info")
                
                tool_output = gr.Textbox(
                    label="Tool Output",
                    lines=10,
                    interactive=False
                )
                
                def test_calculator():
                    try:
                        import subprocess
                        subprocess.run(["open", "-a", "Calculator"], check=True)
                        return "✅ Calculator opened successfully!"
                    except Exception as exc:
                        return f"❌ Error: {exc}"
                
                def show_system_info():
                    try:
                        import subprocess
                        result = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
                        return result.stdout
                    except Exception as exc:
                        return f"❌ Error: {exc}"
                
                test_tool_btn.click(test_calculator, outputs=tool_output)
                system_info_btn.click(show_system_info, outputs=tool_output)
            
            # Settings Tab
            with gr.Tab("⚙️ Settings", id="settings"):
                gr.Markdown("""
                ### Jarvis Configuration
                
                **Current Settings:**
                - Model: `{jarvis.MODEL_NAME}`
                - Ollama Host: `{jarvis.OLLAMA_HOST}`
                - STT Engine: Local Whisper (faster-whisper)
                - TTS Engine: Local (pyttsx3 + macOS say)
                - Compute Type: Auto (optimized for Mac M4)
                """)
                
                with gr.Row():
                    restart_btn = gr.Button("🔄 Restart Jarvis")
                    clear_logs_btn = gr.Button("🗑️ Clear Logs")
                
                settings_output = gr.Textbox(
                    label="Settings Output",
                    lines=5,
                    interactive=False
                )
                
                def restart_jarvis():
                    return "🔄 Restart functionality would be implemented here"
                
                def clear_logs():
                    try:
                        import shutil
                        if os.path.exists("logs"):
                            shutil.rmtree("logs")
                        if os.path.exists("sessions"):
                            shutil.rmtree("sessions")
                        os.makedirs("logs", exist_ok=True)
                        os.makedirs("sessions", exist_ok=True)
                        return "✅ Logs and sessions cleared successfully!"
                    except Exception as exc:
                        return f"❌ Error: {exc}"
                
                restart_btn.click(restart_jarvis, outputs=settings_output)
                clear_logs_btn.click(clear_logs, outputs=settings_output)
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Jarvis Enhanced UI...")
    print("💡 Optimized for Mac M4 with MPS acceleration")
    print("🔒 Fully local - no data leaves your Mac")
    
    ui = create_enhanced_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Keep local for privacy
        show_error=True,
        quiet=False
    )
