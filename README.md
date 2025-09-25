## Jarvis – Local LLM Voice Assistant (macOS‑friendly)

Jarvis is a privacy‑first local assistant that listens, thinks with a local LLM via Ollama, and speaks back using local TTS. It supports push‑to‑talk and continuous listening, optional wake word, and a simple web UI.

### Features
- Fully local: Ollama for LLM, local STT/TTS options
- Push‑to‑talk and continuous modes
- Streaming replies in terminal; simple Gradio UI
- Device selection and robust mic handling
- Optional tools with confirmation (open apps/URLs)
- Logs per session under `logs/`

### Requirements
- macOS (Apple Silicon recommended), Python 3.9+
- Ollama installed and running locally (`ollama serve`)
- Homebrew prerequisites for audio (PortAudio):
```bash
brew install portaudio
pip install -r requirements.txt
```

### Quickstart
```bash
git clone <your-fork-url>
cd HomeBot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 jarvis.py --stream
```

Interactive voice, continuous streaming, local STT and VAD:
```bash
python3 jarvis.py --continuous --stream --stt faster --vad webrtc
```

Simple web UI (single‑turn, non‑streaming baseline):
```bash
python3 -m homebot.ui
```

Tail live transcript logs:
```bash
mkdir -p logs
tail -f logs/$(ls -t logs | head -n1)
```

### Configuration
Environment variables:
```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M
export OLLAMA_TEMPERATURE=0.7
export TTS_RATE_WPM=185
export TTS_VOICE="" # or a system voice name
```

Common CLI flags:
- `--continuous`: background listening without keypresses
- `--stt {google,faster}`: choose STT engine (local `faster` recommended)
- `--vad {off,webrtc}`: enable VAD gating
- `--device-index N` or `--device-name "Substring"`: mic selection
- `--stream`: stream model tokens and speak partial sentences
- `--profile {default,coding,story}`: style preset
- `--enable-tools [--yes]`: allow safe local tool execution

Examples:
```bash
python3 jarvis.py --text "Explain the plan for today" --profile coding
python3 jarvis.py --continuous --stt faster --vad webrtc --device-name "MacBook"
```

### Web UI
`ui.py` provides a minimal Gradio interface:
- Message textbox
- Profile dropdown
- Speak replies (local TTS)

Planned upgrades: streaming tokens, microphone input in the browser, and chat history.

### Voice and Audio
- Baseline TTS: `pyttsx3` with macOS `say` fallback
- Optional neural TTS: Coqui XTTS if installed and `Female2.wav` sample present
- STT: Google fallback; recommended local `faster-whisper` with `--stt faster`

### Troubleshooting
- Mic not found: grant Terminal mic permission; try `--device-index` or `--device-name`
- Ollama errors: `ollama list` then `ollama run <model>`; ensure server running at `OLLAMA_HOST`
- TTS silent: try `say "test"`; switch voices or remove custom VOICE_NAME
- No logs: ensure `logs/` exists and writable

### Roadmap
See `UPGRADE_TASKS.md` for a prioritized senior‑level plan: VAD+Whisper default, wake word, streaming UI with mic, plugin tools, memory/RAG, metrics, tests, and packaging.

### License
MIT (suggested; adjust as desired).

