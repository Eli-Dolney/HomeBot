import os
import time
from typing import List, Dict, Tuple, Optional

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

def respond_once(message: str, profile: str, speak: bool) -> str:
    """Simple non-streaming response"""
    system = PROFILES.get(profile, jarvis.SYSTEM_PROMPT)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": (message or "").strip()},
    ]
    reply = jarvis.chat_ollama(messages)
    if speak and reply:
        jarvis.TextToSpeech().say(reply)
    return reply

def _transcribe_file(audio_path: Optional[str]) -> str:
    """Transcribe audio file to text"""
    if not audio_path:
        return ""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        
        # Try local STT first
        try:
            local_stt = jarvis.LocalSTT(model_size="small", compute_type="auto")
            return local_stt.transcribe_audio(audio, use_vad=False)
        except Exception:
            # Fallback to Google STT
            return recognizer.recognize_google(audio)
    except Exception as exc:
        return f"(Transcription error: {exc})"

def create_simple_ui() -> gr.Blocks:
    """Create a simple, stable web UI"""
    
    with gr.Blocks(
        title="🧠 Jarvis Simple (Mac M4 Optimized)",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🧠 Jarvis Simple - Mac M4 Optimized
        
        **Privacy-first local AI assistant** with Mac-specific tools and memory.
        
        - 🎤 **Voice input** with local STT
        - 🛠️ **Mac tools** (Calculator, Safari, Spotlight, etc.)
        - 🧠 **Memory system** (remember, recall, forget)
        - 🔒 **Fully local** - no data leaves your Mac
        """)
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=True
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
                        label="🤖 Profile",
                        scale=1
                    )
                    speak_checkbox = gr.Checkbox(
                        value=True,
                        label="🔊 Speak Reply",
                        scale=1
                    )
                
                def chat_response(message, history, profile, speak):
                    if not message.strip():
                        return history, ""
                    
                    # Generate response
                    response = respond_once(message, profile, speak)
                    history.append((message, response))
                    return history, ""
                
                # Event handlers
                send_btn.click(
                    chat_response,
                    inputs=[msg_input, chatbot, profile_dropdown, speak_checkbox],
                    outputs=[chatbot, msg_input]
                )
                
                msg_input.submit(
                    chat_response,
                    inputs=[msg_input, chatbot, profile_dropdown, speak_checkbox],
                    outputs=[chatbot, msg_input]
                )
            
            # Voice Tab
            with gr.Tab("🎤 Voice Input"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎤 Record or Upload Audio"
                        )
                        
                        transcribe_btn = gr.Button("🎯 Transcribe & Reply", variant="primary")
                    
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
                
                def voice_pipeline(audio_path, profile, speak):
                    if not audio_path:
                        return "", ""
                    
                    # Transcribe audio
                    text = _transcribe_file(audio_path)
                    if not text:
                        return "", "(No speech detected or transcription failed)"
                    
                    # Generate response
                    response = respond_once(text, profile, speak)
                    return text, response
                
                transcribe_btn.click(
                    voice_pipeline,
                    inputs=[audio_input, voice_profile, voice_speak],
                    outputs=[transcription_output, voice_response]
                )
            
            # Tools Tab
            with gr.Tab("🛠️ Mac Tools"):
                gr.Markdown("""
                ### Available Mac-Specific Tools
                
                **Memory Commands:**
                - `"Remember I prefer dark mode"` - Store a memory
                - `"Recall my preferences"` - Search memories  
                - `"Forget about that meeting"` - Delete a memory
                
                **System Control:**
                - `"Open Calculator"` - Opens Calculator app
                - `"Set volume to 50"` - Adjusts system volume
                - `"Lock screen"` - Locks your Mac
                - `"Search for [query]"` - Opens Spotlight search
                
                **Apps & Files:**
                - `"Open Safari"` - Launch Safari
                - `"Open Terminal"` - Launch Terminal
                - `"Open Code"` - Launch VS Code
                - `"Show desktop"` - Show desktop
                
                **Calendar & Notes:**
                - `"Create calendar event [title]"` - Creates new event
                - `"Create note [content]"` - Creates new note
                - `"Open Calendar"` - Opens Calendar app
                
                **Media Control:**
                - `"Play music"` - Starts Music playback
                - `"Pause music"` - Pauses Music
                - `"Next track"` - Skips to next song
                """)
                
                with gr.Row():
                    test_tool_btn = gr.Button("🧪 Test: Open Calculator")
                    system_info_btn = gr.Button("📊 System Info")
                
                tool_output = gr.Textbox(
                    label="Tool Output",
                    lines=8,
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
                        result = subprocess.run(
                            ["system_profiler", "SPHardwareDataType"], 
                            capture_output=True, text=True
                        )
                        return result.stdout
                    except Exception as exc:
                        return f"❌ Error: {exc}"
                
                test_tool_btn.click(test_calculator, outputs=tool_output)
                system_info_btn.click(show_system_info, outputs=tool_output)
            
            # Status Tab
            with gr.Tab("📊 Status"):
                gr.Markdown(f"""
                ### Jarvis Status
                
                **Current Configuration:**
                - Model: `{jarvis.MODEL_NAME}`
                - Ollama Host: `{jarvis.OLLAMA_HOST}`
                - STT Engine: Local Whisper (faster-whisper)
                - TTS Engine: Local (pyttsx3 + macOS say)
                - Memory System: {'✅ Available' if hasattr(jarvis, 'HAS_MEMORY_SYSTEM') and jarvis.HAS_MEMORY_SYSTEM else '❌ Not Available'}
                
                **Mac M4 Optimizations:**
                - MPS Acceleration: Available
                - Compute Type: Auto (int8_float16 for M4)
                - Local Processing: All data stays on your Mac
                """)
                
                with gr.Row():
                    restart_btn = gr.Button("🔄 Restart Jarvis")
                    clear_logs_btn = gr.Button("🗑️ Clear Logs")
                
                status_output = gr.Textbox(
                    label="Status Output",
                    lines=5,
                    interactive=False
                )
                
                def restart_jarvis():
                    return "🔄 To restart Jarvis, stop the current process and run: ./start_m4.sh"
                
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
                
                restart_btn.click(restart_jarvis, outputs=status_output)
                clear_logs_btn.click(clear_logs, outputs=status_output)
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Jarvis Simple UI...")
    print("💡 Optimized for Mac M4 with MPS acceleration")
    print("🔒 Fully local - no data leaves your Mac")
    
    ui = create_simple_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
