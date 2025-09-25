## Jarvis – Senior-Level Upgrade Plan (Prioritized)

This document lists concrete, high‑impact upgrades to evolve `jarvis.py` into a robust, senior‑quality local assistant. Tasks are grouped by theme and prioritized. Each item has an outcome and brief acceptance criteria.

### 0) Baseline Hardening (P0)
- Logging overhaul: structured JSONL per session
  - Outcome: `sessions/<id>/transcript.jsonl` with {ts, role, text}, `metrics.json` (latencies), and rotating plain text logs for tailing.
  - Accept: Files created per run; tail shows live entries; metrics include STT/LLM/TTS timings.

- Error taxonomy and retries
  - Outcome: Distinguish microphone, STT, LLM connectivity, and TTS failures; implement bounded backoff; never crash process on transient errors.
  - Accept: Simulated failures recover without exiting; user sees actionable messages.

- Config surface consolidation
  - Outcome: Optional `config.yaml` with sane defaults that override env/CLI order: yaml < env < CLI.
  - Accept: Changing a config value is reflected without code edits; help shows precedence.

### 1) Local STT, VAD, Wake Word (P0)
- WebRTC VAD integration in all modes
  - Outcome: VAD gates STT calls; aggressiveness configurable; hangover to avoid truncation.
  - Accept: Background noise no longer triggers STT; speech reliably detected.

- Faster‑Whisper as default STT
  - Outcome: Local STT path is primary with `compute_type=int8_float16` on Apple Silicon; fallback to Google only if explicitly requested.
  - Accept: Offline transcription with good latency; CLI `--stt faster` is default.

- Wake word detector (Porcupine or ONNX KWS)
  - Outcome: Hands‑free activation with refractory period and earcons.
  - Accept: Saying the wake word consistently opens a capture window; no rapid retriggers.

### 2) Streaming UX and Web UI (P1)
- Token streaming in Web UI
  - Outcome: `ui.py` uses generator from `chat_ollama_stream`; partial tokens render progressively.
  - Accept: Visible incremental text; stop button cancels request.

- Mic support in Web UI
  - Outcome: `gr.Audio` input that runs `LocalSTT`; press‑to‑talk and continuous options.
  - Accept: Speak in browser, see transcript and streamed reply locally.

- Barge‑in behavior
  - Outcome: New speech interrupts TTS and model streaming.
  - Accept: Speaking mid‑reply stops audio within <150 ms and defers LLM continuation.

### 3) Memory and Personalization (P1)
- Rolling memory with token budget
  - Outcome: Auto‑truncate conversation to fit model context; optional per‑topic summary messages.
  - Accept: Never exceeds context window; summaries preserve salient facts.

- Long‑term memory (local vector store)
  - Outcome: Add `faiss-cpu` + `sentence-transformers` for remembering facts (`remember <note>`) and retrieving on demand.
  - Accept: `remember` stores; related questions are augmented with retrieved snippets.

### 4) Tools Framework (P1)
- Pluggable tool system with allowlists
  - Outcome: YAML manifests define tools (name, params, risks); runtime asks consent unless `--yes`.
  - Accept: Adding a new tool requires no core code change; dry‑run shows command.

- Example local tools
  - Outcome: Open apps/URLs; file search; calendar/notes via AppleScript; optional Home Assistant local API.
  - Accept: Demos work across a few representative tasks with clear prompts and results.

### 5) TTS Quality & Voices (P2)
- High‑quality local TTS
  - Outcome: Integrate Piper or improve Coqui XTTS path with device selection and preloading.
  - Accept: Lower latency and better prosody vs `pyttsx3`.

- Voice profiles and rate presets
  - Outcome: Quick toggles (natural, fast, narration); VOICE_NAME discovery utility.
  - Accept: Switching profiles updates voice/rate instantly.

### 6) Observability & Tests (P2)
- Metrics and health panel
  - Outcome: `rich` status showing mic, VAD, STT/LLM/TTS timings, CPU/GPU utilization hints.
  - Accept: Panel updates per interaction; anomalies highlighted.

- Test suite
  - Outcome: Unit tests for prompt/memory; integration tests on golden audio; regression guard for latency.
  - Accept: `pytest` green locally; CI runs basic checks.

### 7) Packaging & Operations (P2)
- CLI polish
  - Outcome: `--help` with examples; subcommands (`chat`, `listen`, `serve-ui`).
  - Accept: Clear discoverability; sensible defaults.

- macOS LaunchAgent / Menu bar
  - Outcome: Optional background auto‑start and quick toggle UI.
  - Accept: Assistant can start on login and be paused easily.

### 8) Security & Privacy (Ongoing)
- Explicit offline mode (default)
  - Outcome: No outbound network calls except `OLLAMA_HOST`; toggle for any future connectors.
  - Accept: Network inspector shows local‑only during normal operation.

- Data retention controls
  - Outcome: `--no-log` and `scripts/purge.sh` to remove all sessions.
  - Accept: Running purge leaves no residual PII in project data dirs.

---

### Milestone Breakdown
- v0.2 (Core Local): VAD+Whisper default, wake word, JSONL logs.
- v0.3 (UX & Tools): Streaming UI, barge‑in, plugin tools, metrics panel.
- v0.4 (Quality): Neural TTS, memory/RAG, test suite and CI.

### Notes
- Favor Apple Silicon paths (`mps`) where possible; fallbacks for CPU.
- Keep destructive tools opt‑in with explicit confirmations and dry‑runs.

