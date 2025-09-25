## All‑Local Jarvis Assistant – Senior Design Document

### 1) Purpose and Scope
Build an all‑local, privacy‑preserving Jarvis‑style voice assistant that:
- Runs fully on the user’s machine (no cloud calls).
- Listens for user speech, converts to text (STT), reasons with a local LLM via Ollama, and replies via TTS.
- Supports push‑to‑talk and continuous modes with wake‑word + VAD.
- Offers a modular, extensible “tools” interface (e.g., open apps, control smart devices via local bridges) with strong safety/consent.

Non‑goals (initial phase):
- Cloud connectivity, account management, web UIs. Focus is local CLI/daemon.
- Advanced multi‑speaker diarization. Single user primary.


### 2) User Stories
- As a user, I can press a key or say a wake word to speak, and the assistant will hear me while ignoring background noise.
- As a user, I receive natural‑sounding spoken replies quickly.
- As a user, I can operate fully offline.
- As a user, I can enable/disable continuous listening and choose the input/output devices.
- As a user, I can extend the assistant with local “tools” (e.g., shell commands allowlisted per task).
- As a user, I can see transcripts and turn history saved locally.


### 3) System Overview
High‑level flow:
1. Audio Capture → VAD + optional Wake Word → STT → LLM Prompting → TTS.
2. The Orchestrator manages modes, memory, tool invocation, and error handling.

Textual sequence (continuous mode):
1) Microphone stream → (optional) Wake Word detector → if detected, open a capture window.
2) VAD segments speech → STT transcribes utterance.
3) Orchestrator builds messages with system + memory + user text → sends to Ollama (local) → receives assistant text.
4) TTS speaks reply; optionally stream partials if available.
5) Transcripts and metadata are appended to local storage.


### 4) Components

#### 4.1 Orchestrator (Python)
- Responsibilities: lifecycle, configuration, mode switching (push‑to‑talk vs continuous), memory, prompt assembly, tool routing, logging.
- Interfaces: STT, VAD/WakeWord, LLM Client, TTS, Tools, Storage.
- Error policy: recover without exiting; exponential backoff on transient failures.

#### 4.2 Audio Capture
- Implementation: PyAudio (PortAudio) for cross‑platform capture.
- Device selection by index or substring; robust retry and listing on failure.
- Sample rate: 16 kHz mono for STT; buffer sizes tuned to latency targets (e.g., 20–30 ms frames).

#### 4.3 Voice Activity Detection (VAD)
- Goal: reject background noise; gate STT with speech segments.
- Options:
  - WebRTC VAD (py-webrtcvad): CPU‑light, stable; classify 10/20/30 ms frames.
  - Silero VAD (onnx): more accurate in noise; slightly higher CPU.
- Configuration: aggressiveness level (0–3 for WebRTC); hangover time; min/max utterance length.

#### 4.4 Wake Word (optional but recommended)
- Options: Porcupine (local, commercial license), open‑keyword models (Picovoice custom, open KWS), or a tiny onnx model.
- Design: run wake detector on low‑rate audio; on trigger, enable ASR window + earcon.
- Safety: suppress re‑triggers using refractory period.

#### 4.5 Speech‑to‑Text (STT)
- Requirements: fully local, robust, low‑latency.
- Options:
  - whisper.cpp (C++): Tiny/Small/Medium models; great local performance; bindings via python‑whispercpp.
  - faster‑whisper (CTranslate2): good perf on CPU/GPU; Pythonic API.
- Recommended baseline: faster‑whisper with `small` for responsiveness on Apple Silicon; upgrade to `medium` if CPU/GPU allows.
- Features: chunked streaming decode (if available), endpointing from VAD, fallback to push‑to‑talk if noisy.

#### 4.6 LLM (Reasoning)
- Engine: Ollama hosting `gemma3:4b` (present) with option to swap models per task.
- Prompting: concise system prompt; summaries for long contexts; configurable temperature and top‑p.
- Memory: rolling conversation window with token budget management; optional long‑term store (vector DB) for facts.
- Safety: tool execution gated by explicit user confirmation unless pre‑approved.

#### 4.7 Text‑to‑Speech (TTS)
- Baseline: `pyttsx3` (system voices), with macOS `say` fallback.
- Optional: neural local TTS (e.g., Coqui‑TTS or Piper) for higher quality voices.
- Controls: rate, voice name, volume, and barge‑in (interrupt playback when user starts speaking).

#### 4.8 Tools (Extensibility)
- Local tool adapters with strict allowlists and prompts:
  - App control (open/close apps), filesystem queries, local HTTP calls.
  - Home automation via local bridges (e.g., Home Assistant local API, Matter bridges).
- Policy:
  - Tools declare parameters and side‑effects.
  - Orchestrator requires user consent for destructive actions.
  - Dry‑run preview before execution.

#### 4.9 Storage & Telemetry (Local Only)
- Transcripts: JSONL per session with timestamps, device, thresholds, tokens.
- Audio snippets (optional, off by default).
- Metrics: latency per stage, word error rate (if labeled tests), CPU/GPU usage.
- Privacy: store under user directory; easy purge command.


### 5) Configuration
- Env vars: `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TEMPERATURE`, `TTS_RATE_WPM`, `TTS_VOICE`.
- CLI flags: `--continuous`, `--text ...`, `--device-index`, `--device-name`, `--vad {webrtc,silero}`, `--wake {off,porcupine,onnx}`.
- Config file: `config.yaml` (optional) for persistent defaults.


### 6) Performance Targets
- Cold start < 2s (excluding model load on first Ollama run).
- End‑to‑end latency (end of speech → first spoken token) < 1.0s on Apple Silicon for small models.
- Continuous mode CPU: < 60% on M‑series Macs; memory < 4 GB incremental.


### 7) Security & Privacy
- All inference is local. No outbound network by default (besides Ollama localhost).
- Tools run in a restricted mode with explicit allowlists.
- PII handling: local only; redact optional; easy “panic purge” of all logs.
- Mic access: visible indicator + on/off control; wake‑word model runs locally only.


### 8) Failure Modes & Recovery
- Mic unavailable → list devices, retry with backoff, keep process alive.
- STT timeouts → prompt user to retry; adjust VAD thresholds dynamically.
- LLM server down → auto‑retry, surface status; optional local queueing.
- TTS failure → fall back to alternate engine; always print text.


### 9) Testing Strategy
- Unit tests:
  - Prompt builder, memory truncation, tool gating logic.
  - Audio framing → VAD segmentation on synthetic audio.
- Integration tests:
  - Golden audio files → expected transcripts.
  - Loop test: STT → LLM → TTS on a fixed script.
- Performance tests:
  - Measure latency/CPU across models; store baselines.
- Safety tests:
  - Tool command approval flows; denial and dry‑run cases.


### 10) Implementation Plan (Phased)
Phase 1 (MVP: already started):
- CLI modes (push‑to‑talk, `--continuous`).
- Robust mic device selection, retry loop, ambient calibration.
- LLM via Ollama (`gemma3:4b`), TTS via `pyttsx3` with `say` fallback.

Phase 2 (All‑Local Core):
- Integrate VAD (WebRTC VAD) and swap STT to local (faster‑whisper or whisper.cpp).
- Add wake‑word (Porcupine or ONNX KWS) and earcons.
- Barge‑in: stop TTS when user starts speaking.
- Structured logging (JSONL) and session storage.

Phase 3 (Tools & Safety):
- Tool plugin framework with YAML allowlists and user confirmation UI in CLI.
- Example tools: open URL, launch apps, query Home Assistant locally.
- Safety prompts and dry‑runs.

Phase 4 (Quality & Tuning):
- Neural TTS (Piper/Coqui) and voice profiles.
- Latency tuning: streaming decode in STT; partial LLM streaming.
- Adaptive thresholds for noisy environments.


### 11) Detailed Design Notes

#### 11.1 Orchestrator
- State machine: Idle → (Wake/PTT) → Listening → Transcribing → Thinking → Speaking → Idle.
- Memory manager: token‑budgeted window; optional summaries every N turns.
- Prompting:
  - System: concise, voice‑friendly style guidelines.
  - Context injection: device profile, location (optional), tool schemas.

#### 11.2 STT Pipeline
- Pre‑emphasis, optional noise suppression (RNNoise/NSNet2) before VAD.
- VAD frames (20 ms), hangover 200–400 ms, max utterance 15 s.
- Whisper inference with beam/temperature tuned for speed.
- Endpointing harmonized with VAD to prevent double truncation.

#### 11.3 LLM API (Ollama)
- Use chat API; prefer streaming (if switching to partial playback later).
- Options (temperature, top‑p) exposed via config and CLI.
- Error handling: distinguish connection vs model errors; suggest `ollama run` when absent.

#### 11.4 TTS Subsystem
- Abstraction supports multiple engines.
- Barge‑in implemented via an “is‑speaking” flag and stop method; integrate with `say` or neural TTS APIs that expose stop.
- Rate and voice selection from config; test voices enumerated at boot.

#### 11.5 Tools
- Each tool: schema (name, params, risks), permission level, implementation.
- Call protocol: LLM proposes tool call → Orchestrator validates + seeks explicit consent → executes → returns result text back as context.
- Logging: inputs/outputs stored locally; redact secrets.

#### 11.6 Storage
- `sessions/SESSION_ID/` with `transcript.jsonl`, `metrics.json`, optional `audio/`.
- Helper scripts: `scripts/purge.sh`, `scripts/analyze_metrics.py`.


### 12) Dependencies & Runtime
- Required now: Python 3.9+, PortAudio (Homebrew), PyAudio, SpeechRecognition (for fallback), pyttsx3.
- Planned local STT/VAD:
  - `webrtcvad` (VAD)
  - `faster-whisper` (CTranslate2) or `whispercpp` bindings
  - Optional: `piper-tts` or `TTS` (Coqui)
- Ollama with `gemma3:4b` (already installed).


### 13) Configuration Examples
Env:
```
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gemma3:4b
export OLLAMA_TEMPERATURE=0.7
export TTS_RATE_WPM=185
export TTS_VOICE=""
```

CLI:
```
python jarvis.py --continuous --device-name "MacBook Air Microphone"
python jarvis.py --device-index 2
python jarvis.py --text "What’s the weather offline style?"
```


### 14) Operational Runbook
- If no audio heard: check mic permissions (System Settings → Privacy & Security → Microphone), use `--device-index`, verify with `pyaudio` device listing.
- If LLM fails: `ollama list`, `ollama run gemma3:4b`, restart daemon.
- If TTS silent: try `say "test"`; switch engine or voice.
- Purge data: delete `sessions/` directory.


### 15) Roadmap Summary
- v0.1: Current CLI with robust mic handling, Ollama LLM, TTS with fallback.
- v0.2: WebRTC VAD + faster‑whisper local STT; wake‑word; earcons; logs.
- v0.3: Tools framework with safety; local home automation adapters.
- v0.4: Neural TTS, streaming LLM/STT for sub‑second responses.


### 16) Acceptance Criteria (MVP → v0.2)
- Works offline; no external API network calls.
- Continuous mode runs > 1 hour without exit; background noise ignored.
- STT accuracy >= Whisper small baseline on test set; median E2E latency < 1.2s.
- TTS always audible or falls back to text + alternate engine.


### 17) Open Questions
- Preferred wake‑word and licensing constraints?
- Should we store audio by default (privacy trade‑off vs debugging)?
- Do we need multilingual STT/TTS by default or opt‑in?
