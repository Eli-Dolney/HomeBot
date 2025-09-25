from typing import List, Dict, Any, Optional, Generator

def chat_ollama_stream(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> Generator[str, None, None]:
	"""Yield chunks of assistant text from Ollama streaming API."""
	# Late-bind defaults to avoid NameError at import time
	if model is None:
		model = MODEL_NAME
	if temperature is None:
		temperature = TEMPERATURE
	url = f"{OLLAMA_HOST}/api/chat"
	payload: Dict[str, Any] = {
		"model": model,
		"messages": messages,
		"stream": True,
		"keep_alive": "1h",
		"options": {"temperature": temperature, "num_predict": 256},
	}
	with requests.post(url, json=payload, stream=True, timeout=120) as resp:
		resp.raise_for_status()
		for line in resp.iter_lines(decode_unicode=True):
			if not line:
				continue
			try:
				data = json.loads(line)
			except Exception:
				continue
			msg = data.get("message", {})
			content = (msg.get("content") or "")
			if content:
				yield content

import os
import sys
import time
import json
import shutil
import subprocess
import argparse
from typing import List, Dict, Any, Optional
import tempfile
try:
	import torch  # type: ignore
except Exception:
	torch = None  # type: ignore

import requests
import speech_recognition as sr
import numpy as np
from rich.console import Console
from rich.panel import Panel
import shlex
import re
import yaml

# Import memory system
try:
	from memory_system import get_memory_system, remember, recall, forget
	HAS_MEMORY_SYSTEM = True
except ImportError:
	HAS_MEMORY_SYSTEM = False
	print("Memory system not available. Install with: pip install sentence-transformers faiss-cpu")

# Session logging (logs/ and sessions/ folders)
LOGS_DIR = os.environ.get("JARVIS_LOG_DIR", "logs")
SESSIONS_ROOT = os.environ.get("JARVIS_SESSIONS_DIR", "sessions")
try:
	os.makedirs(LOGS_DIR, exist_ok=True)
except Exception:
	pass
try:
	os.makedirs(SESSIONS_ROOT, exist_ok=True)
except Exception:
	pass
SESSION_LOG = os.path.join(LOGS_DIR, f"jarvis_{int(time.time())}.log")

# These are initialized in main() when a session starts
SESSION_DIR: Optional[str] = None
TRANSCRIPT_JSONL: Optional[str] = None
METRICS_JSONL: Optional[str] = None

def _append_jsonl(path: Optional[str], obj: Dict[str, Any]) -> None:
	if not path:
		return
	try:
		with open(path, "a", encoding="utf-8") as f:
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")
	except Exception:
		pass

def append_log(role: str, text: str) -> None:
	# Human-readable rolling text log
	try:
		with open(SESSION_LOG, "a", encoding="utf-8") as f:
			f.write(f"[{time.strftime('%H:%M:%S')}] {role}: {text}\n")
	except Exception:
		pass
	# Structured JSONL transcript per session
	try:
		_append_jsonl(
			TRANSCRIPT_JSONL,
			{
				"ts": time.time(),
				"role": role,
				"text": text,
			},
		)
	except Exception:
		pass

def _init_session_dirs() -> None:
	global SESSION_DIR, TRANSCRIPT_JSONL, METRICS_JSONL
	try:
		session_id = int(time.time())
		session_dir = os.path.join(SESSIONS_ROOT, f"session_{session_id}")
		os.makedirs(session_dir, exist_ok=True)
		SESSION_DIR = session_dir
		TRANSCRIPT_JSONL = os.path.join(session_dir, "transcript.jsonl")
		METRICS_JSONL = os.path.join(session_dir, "metrics.jsonl")
	except Exception:
		SESSION_DIR = None
		TRANSCRIPT_JSONL = None
		METRICS_JSONL = None

try:
	import webrtcvad  # type: ignore
	except_importing_webrtcvad = None
except Exception as exc:
	except_importing_webrtcvad = exc
	webrtcvad = None  # type: ignore

try:
	from faster_whisper import WhisperModel  # type: ignore
	except_importing_fw = None
except Exception as exc:
	except_importing_fw = exc
	WhisperModel = None  # type: ignore

try:
	import pyttsx3
	except_importing_pyttsx3 = None
except Exception as exc:
	except_importing_pyttsx3 = exc
	pyttsx3 = None  # type: ignore

# Optional local TTS voice cloning (Coqui XTTS)
try:
	from TTS.api import TTS as CoquiTTS  # type: ignore
	except_importing_coqui = None
except Exception as exc:
	except_importing_coqui = exc
	CoquiTTS = None  # type: ignore


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))
SPEECH_RATE_WPM = int(os.environ.get("TTS_RATE_WPM", "185"))
VOICE_NAME = os.environ.get("TTS_VOICE", "")  # leave empty for default voice
VOICE_SAMPLE_PATH = os.environ.get("VOICE_SAMPLE_PATH", "Female2.wav")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "en")

SYSTEM_PROMPT = (
	"You are Jarvis, a concise, helpful voice assistant. "
	"Keep responses short and conversational unless asked for detail."
)

CODING_ASSISTANT_SYSTEM = (
	"You are a senior coding assistant co-pilot. "
	"Give concise, actionable answers with examples when helpful. "
	"Prefer clarity over cleverness; point out pitfalls and edge cases."
)

STORYTELLER_SYSTEM = (
	"You are a warm, imaginative storyteller for a young child. "
	"Use simple language, gentle tone, and uplifting themes."
)


class TextToSpeech:
	"""Text-to-speech helper using pyttsx3 or macOS say. Tries local voice cloning if available."""

	def __init__(self) -> None:
		self.engine = None
		self.is_available = False
		self.has_say = shutil.which("say") is not None
		self.is_speaking = False
		self.last_speak_end: float = 0.0
		# Coqui XTTS lazy init flags
		self.has_xtts = False
		self._xtts = None
		# Optimize device selection for Mac M4
		if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
			self._xtts_device = "mps"  # Use Metal Performance Shaders on M4
		else:
			self._xtts_device = "cpu"
		# Enable XTTS if library is present and voice sample exists
		if CoquiTTS is not None and VOICE_SAMPLE_PATH and os.path.exists(VOICE_SAMPLE_PATH):
			self.has_xtts = True
		# Initialize pyttsx3 if available
		if pyttsx3 is not None:
			try:
				self.engine = pyttsx3.init()
				# Configure voice and rate
				if VOICE_NAME:
					for voice in self.engine.getProperty("voices"):
						if VOICE_NAME.lower() in (voice.name or "").lower():
							self.engine.setProperty("voice", voice.id)
							break
				self.engine.setProperty("rate", SPEECH_RATE_WPM)
				self.is_available = True
			except Exception:
				self.engine = None
				self.is_available = False

	def stop(self) -> None:
		try:
			if self.engine is not None:
				self.engine.stop()
		except Exception:
			pass

	def _ensure_xtts(self) -> None:
		"""Load Coqui XTTS model if not yet loaded."""
		if not self.has_xtts or self._xtts is not None:
			return
		try:
			# Multilingual voice cloning model
			self._xtts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2") if CoquiTTS is not None else None
			if self._xtts is not None:
				self._xtts.to(self._xtts_device)
		except Exception:
			self._xtts = None
			self.has_xtts = False

	def say(self, text: str) -> None:
		if not text:
			return
		print(f"Jarvis: {text}")
		append_log("assistant", text)
		# Prefer high-quality local cloning if configured
		if self.has_xtts:
			self._ensure_xtts()
			if self._xtts is not None and VOICE_SAMPLE_PATH and os.path.exists(VOICE_SAMPLE_PATH):
				try:
					with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
						out_path = tmp.name
					# Generate to file and play via macOS afplay (local, no internet)
					self._xtts.tts_to_file(
						text=text,
						speaker_wav=VOICE_SAMPLE_PATH,
						language=TTS_LANGUAGE,
						file_path=out_path,
					)
					subprocess.run(["afplay", out_path], check=False)
					self.last_speak_end = time.time()
					return
				except Exception:
					# Fall through to other engines on error
					pass
		# Prefer pyttsx3 when available
		if self.is_available and self.engine is not None:
			try:
				self.is_speaking = True
				self.engine.say(text)
				self.engine.runAndWait()
				self.last_speak_end = time.time()
			finally:
				self.is_speaking = False
		# macOS 'say' fallback
		elif self.has_say:
			try:
				self.is_speaking = True
				# Clean text to avoid command injection and special characters
				clean_text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text).strip()
				if not clean_text:
					return
				cmd = ["say", "-r", str(SPEECH_RATE_WPM)]
				if VOICE_NAME:
					cmd += ["-v", VOICE_NAME]
				cmd += [clean_text]
				subprocess.run(cmd, check=False, timeout=30)
				self.last_speak_end = time.time()
			except Exception as exc:
				print(f"TTS error: {exc}")
			finally:
				self.is_speaking = False


def _recognize_google(recognizer: sr.Recognizer, audio: sr.AudioData) -> str:
	"""Typed wrapper to access recognizer.recognize_google without pyright errors."""
	return getattr(recognizer, "recognize_google")(audio)


def transcribe_from_mic(recognizer: sr.Recognizer, microphone: sr.Microphone, timeout: float = 6.0, phrase_time_limit: float = 15.0) -> str:
	"""Listen from microphone and transcribe using Google's free recognizer.

	Note: Requires internet. If you want fully local STT, integrate Whisper.
	"""
	with microphone as source:
		print("Listening... (speak now)")
		recognizer.adjust_for_ambient_noise(source, duration=0.4)  # type: ignore[arg-type]
		audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

	try:
		return _recognize_google(recognizer, audio)
	except sr.UnknownValueError:
		return ""
	except sr.RequestError as exc:
		print(f"STT request error: {exc}")
		return ""


def vad_gate_samples(samples: np.ndarray, sample_rate: int, aggressiveness: int = 2) -> bool:
	"""Return True if speech likely present using WebRTC VAD on a short window."""
	if webrtcvad is None:
		return True
	vad = webrtcvad.Vad(aggressiveness)
	frame_ms = 30
	frame_len = int(sample_rate * frame_ms / 1000)
	if len(samples) < frame_len:
		return False
	frame = samples[:frame_len]
	pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
	return vad.is_speech(pcm16, sample_rate)


class LocalSTT:
	"""Local STT using faster-whisper. Lazy model load."""

	model: Any

	def __init__(self, model_size: str = "small", compute_type: str = "auto") -> None:
		self.model_size = model_size
		self.model = None
		# Auto-detect best compute type for Mac M4
		if compute_type == "auto":
			if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
				self.compute_type = "float16"  # Compatible for M4 with MPS
			else:
				self.compute_type = "float32"  # Fallback for CPU
		else:
			self.compute_type = compute_type

	def ensure_model(self) -> None:
		if self.model is None:
			if WhisperModel is None:
				raise RuntimeError("faster-whisper not available")
			# Try preferred compute type first; fall back to safer options if not supported
			preferred_types = [self.compute_type]
			# Add progressively more compatible fallbacks
			if "float16" not in preferred_types:
				preferred_types.append("float16")
			if "float32" not in preferred_types:
				preferred_types.append("float32")
			last_error: Optional[Exception] = None
			for ct in preferred_types:
				try:
					self.model = WhisperModel(self.model_size, device="auto", compute_type=ct)
					self.compute_type = ct
					last_error = None
					break
				except Exception as exc:
					last_error = exc
			if self.model is None:
				raise RuntimeError(f"Failed to initialize faster-whisper with compute types {preferred_types}: {last_error}")

	def transcribe_audio(self, audio: sr.AudioData, use_vad: bool) -> str:
		self.ensure_model()
		assert self.model is not None
		sample_rate = 16000
		raw = audio.get_raw_data(convert_rate=sample_rate, convert_width=2)
		if not raw:
			return ""
		samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
		# Basic silence/rms check to avoid NaNs in feature extraction
		if samples.size < int(sample_rate * 0.1):  # <100ms
			return ""
		rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
		if rms < 1e-4:
			return ""
		# VAD gating: check multiple frames across the first ~1s
		if use_vad:
			frame_ms = 30
			frame_len = int(sample_rate * frame_ms / 1000)
			window = samples[: sample_rate] if samples.size > sample_rate else samples
			any_speech = False
			for start in range(0, len(window) - frame_len + 1, frame_len):
				if vad_gate_samples(window[start : start + frame_len], sample_rate):
					any_speech = True
					break
			if not any_speech:
				return ""
		segments, _ = self.model.transcribe(
			samples,
			language="en",
			task="transcribe",
			vad_filter=use_vad,
			beam_size=1,
			no_speech_threshold=0.6,
		)
		text_parts = [seg.text for seg in segments]
		return (" ".join(text_parts)).strip()


def chat_ollama(messages: List[Dict[str, str]], model: str = MODEL_NAME, temperature: float = TEMPERATURE) -> str:
	"""Send a chat request to Ollama and return the assistant's text."""
	# Enhance messages with relevant memories if available
	if HAS_MEMORY_SYSTEM and messages:
		user_message = messages[-1].get("content", "") if messages else ""
		if user_message and user_message.strip():
			# Retrieve relevant memories
			relevant_memories = recall(user_message, limit=3)
			if relevant_memories:
				# Add memory context to the conversation
				memory_context = "Relevant memories:\n"
				for memory in relevant_memories:
					memory_context += f"- {memory.content}\n"
				memory_context += "\n"
				
				# Insert memory context before the user message
				if len(messages) > 1:
					messages.insert(-1, {"role": "system", "content": memory_context})
				else:
					messages.insert(0, {"role": "system", "content": memory_context})
	
	url = f"{OLLAMA_HOST}/api/chat"
	payload: Dict[str, Any] = {
		"model": model,
		"messages": messages,
		"stream": False,
		"keep_alive": "1h",
		"options": {"temperature": temperature, "num_predict": 256},
	}
	try:
		t0 = time.time()
		resp = requests.post(url, json=payload, timeout=120)
		resp.raise_for_status()
		data = resp.json()
		# Expected shape: { message: { role: "assistant", content: "..." }, ... }
		msg = data.get("message", {})
		content = (msg.get("content") or "").strip()
		t1 = time.time()
		# Record latency metric
		_append_jsonl(METRICS_JSONL, {"ts": t1, "stage": "llm", "duration_sec": t1 - t0})
		return content
	except requests.RequestException as exc:
		return f"(Network error talking to Ollama: {exc})"
	except json.JSONDecodeError:
		return "(Invalid JSON from Ollama)"


def main() -> None:
	# CLI args
	parser = argparse.ArgumentParser(description="Jarvis voice assistant (Ollama)")
	parser.add_argument("--text", nargs="*", help="Send text instead of using mic")
	parser.add_argument("--continuous", action="store_true", help="Continuous listening (no keypress needed)")
	parser.add_argument("--stt", choices=["google", "faster"], default="faster", help="Choose STT engine")
	parser.add_argument("--vad", choices=["off", "webrtc"], default="webrtc", help="Enable VAD gating")
	parser.add_argument("--vad-aggr", type=int, default=2, help="WebRTC VAD aggressiveness (0-3)")
	parser.add_argument("--whisper-model", default="small", help="faster-whisper model size (tiny, base, small, medium)")
	parser.add_argument("--stt-compute", choices=["auto", "float32", "float16", "int8", "int8_float16"], default="auto", help="CT2 compute type for faster-whisper (auto=optimized for Mac M4)")
	parser.add_argument("--stream", action="store_true", help="Stream model tokens and speak partial sentences")
	parser.add_argument("--profile", choices=["default", "coding", "story"], default="default", help="Assistant style profile")
	parser.add_argument("--device-index", type=int, default=None, help="Microphone device index to use")
	parser.add_argument("--device-name", type=str, default=None, help="Substring of microphone device name to match")
	parser.add_argument("--enable-tools", action="store_true", help="Allow limited local tool execution with confirmation")
	parser.add_argument("--yes", action="store_true", help="Auto-confirm tool execution")
	parser.add_argument("--tools-config", type=str, default="tools.yaml", help="Path to tools allowlist YAML")
	parser.add_argument("--wake", choices=["off", "scaffold", "stt"], default="off", help="Wake-word mode: 'stt' uses STT to detect the word 'jarvis'")
	args = parser.parse_args()

	# Initialize structured session directory
	_init_session_dirs()

	# Allow quick text test without mic: `python jarvis.py --text Hello there`
	if args.text:
		user_text = " ".join(args.text).strip() or "Hello"
		profile_prompt = SYSTEM_PROMPT
		if args.profile == "coding":
			profile_prompt = CODING_ASSISTANT_SYSTEM
		elif args.profile == "story":
			profile_prompt = STORYTELLER_SYSTEM
		text_mode_messages: List[Dict[str, str]] = [
			{"role": "system", "content": profile_prompt},
			{"role": "user", "content": user_text},
		]
		reply = chat_ollama(text_mode_messages)
		print(reply)
		TextToSpeech().say(reply)
		return

	# Voice loop
	tts = TextToSpeech()
	console = Console()
	if args.wake == "scaffold":
		console.print(Panel.fit("Wake-word not implemented yet; using current mode.", title="Wake", style="yellow"))

	def maybe_run_tool(user_text: str) -> Optional[str]:
		if not args.enable_tools:
			return None
		
		# Handle memory commands first
		if HAS_MEMORY_SYSTEM:
			text = (user_text or "").strip().lower()
			
			# Remember command
			if text.startswith("remember "):
				content = user_text[9:].strip()  # Remove "remember "
				if content:
					memory_id = remember(content, memory_type="fact", importance=0.7)
					return f"I'll remember: {content}"
			
			# Recall command
			elif text.startswith("recall "):
				query = user_text[7:].strip()  # Remove "recall "
				if query:
					memories = recall(query, limit=5)
					if memories:
						result = f"Here's what I remember about '{query}':\n"
						for i, memory in enumerate(memories, 1):
							result += f"{i}. {memory.content}\n"
						return result
					else:
						return f"I don't have any memories about '{query}'"
			
			# Forget command
			elif text.startswith("forget "):
				query = user_text[7:].strip()  # Remove "forget "
				if query:
					memories = recall(query, limit=10)
					if memories:
						# Delete the most relevant memory
						forget(memories[0].id)
						return f"I've forgotten: {memories[0].content}"
					else:
						return f"I don't have any memories about '{query}' to forget"
		
		# Load tools manifest lazily
		tools_manifest: Dict[str, Any] = {}
		try:
			if os.path.exists(args.tools_config):
				with open(args.tools_config, "r", encoding="utf-8") as f:
					tools_manifest = yaml.safe_load(f) or {}
		except Exception as exc:
			return f"(Failed to load tools config: {exc})"
		patterns = tools_manifest.get("patterns", [])
		if not isinstance(patterns, list):
			return None
		text = (user_text or "").strip()
		for entry in patterns:
			name = (entry or {}).get("name")
			match = (entry or {}).get("match")
			command = (entry or {}).get("command")
			args_template = (entry or {}).get("args", [])
			if not (name and match and command):
				continue
			try:
				if match.startswith("startswith:"):
					prefix = match.split(":", 1)[1].strip()
					if not text.lower().startswith(prefix.lower()):
						continue
					remainder = text[len(prefix):].strip()
				elif match.startswith("equals:"):
					needle = match.split(":", 1)[1].strip().lower()
					if text.lower() != needle:
						continue
					remainder = ""
				else:
					continue
				cmd: list[str] = [command]
				for a in args_template:
					cmd.append(a.replace("{remainder}", remainder))
				# Ask for consent
				if not args.yes and sys.stdin.isatty():
					confirm = input(f"Run tool '{name}': {' '.join(cmd)} ? [y/N]: ").strip().lower()
					if confirm not in ("y", "yes"):
						console.print(Panel.fit("Cancelled.", title="Tool", style="yellow"))
						return "Cancelled"
				subprocess.run(cmd, check=True)
				return f"Tool '{name}' executed successfully"
			except subprocess.CalledProcessError as exc:
				return f"Execution failed (exit {exc.returncode})"
			except Exception as exc:
				return f"Tool error: {exc}"
		return None
	recognizer = sr.Recognizer()
	# Tune recognizer to be less sensitive to background noise
	recognizer.dynamic_energy_threshold = True
	recognizer.dynamic_energy_adjustment_damping = 0.15
	recognizer.dynamic_energy_ratio = 2.0  # be stricter about what counts as speech
	recognizer.pause_threshold = 0.7  # shorter pause to end phrase
	recognizer.phrase_threshold = 0.3  # require brief speech before starting
	recognizer.non_speaking_duration = 0.3  # shorter leading/trailing silence
	def resolve_device_index() -> Optional[int]:
		if args.device_index is not None:
			return args.device_index
		if args.device_name:
			try:
				available = sr.Microphone.list_microphone_names()
				for idx, name in enumerate(available):
					if args.device_name.lower() in (name or "").lower():
						return idx
			except Exception:
				pass
		return None

	# Robust microphone acquisition with retries instead of exiting
	microphone = None  # type: ignore
	attempt = 0
	while microphone is None:
		attempt += 1
		try:
			device_idx = resolve_device_index()
			microphone = sr.Microphone(device_index=device_idx)
		except Exception as exc:
			print("Could not access microphone. If on macOS, grant mic permission to Terminal/IDE.")
			print(f"Details: {exc}")
			try:
				available = sr.Microphone.list_microphone_names()
				print("Available input devices:")
				for i, name in enumerate(available):
					print(f"  [{i}] {name}")
			except Exception:
				print("(Could not list microphones)")
			if sys.stdin.isatty():
				try:
					choice = input("Enter device index to use (or press Enter to retry): ").strip()
					if choice:
						args.device_index = int(choice)
				except Exception:
					pass
			else:
				print("Retrying microphone access in 5s...")
				time.sleep(5)

	# Profile selection
	profile_prompt = SYSTEM_PROMPT
	if args.profile == "coding":
		profile_prompt = CODING_ASSISTANT_SYSTEM
	elif args.profile == "story":
		profile_prompt = STORYTELLER_SYSTEM

	messages: List[Dict[str, str]] = [
		{"role": "system", "content": profile_prompt},
	]

	# Extended ambient noise calibration at startup
	try:
		with microphone as source:
			print("Calibrating for ambient noise... (1.5s)")
			recognizer.adjust_for_ambient_noise(source, duration=1.5)  # type: ignore[arg-type]
			# Raise the threshold a bit above measured ambient to avoid background activations
			baseline = recognizer.energy_threshold
			recognizer.energy_threshold = max(350, int(baseline * 1.5))
			print(f"Calibrated energy threshold: {recognizer.energy_threshold}")
	except Exception:
		pass

	# Optional continuous mode avoids needing a keyboard/TTY
	local_stt = LocalSTT(args.whisper_model, compute_type=args.stt_compute) if args.stt == "faster" else None

	if args.continuous:
		last_text: List[str] = [""]
		last_time: List[float] = [0.0]
		def _bg_callback(rec: sr.Recognizer, audio: sr.AudioData) -> None:
			try:
				if args.stt == "faster" and local_stt is not None:
					text = local_stt.transcribe_audio(audio, use_vad=(args.vad == "webrtc"))
				else:
					text = _recognize_google(rec, audio)
			except sr.UnknownValueError:
				return
			except Exception as exc:
				print(f"STT error: {exc}")
				return
			if not text:
				return
			# Deduplicate near-identical/echoed transcripts within 2 seconds
			now = time.time()
			curr = text.strip()
			prev = last_text[0].strip()
			# remove simple repeated phrases like "hello hello" → "hello"
			collapsed = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", curr, flags=re.IGNORECASE)
			text = collapsed
			if curr.lower() == prev.lower() and (now - last_time[0]) < 2.0:
				return
			last_text[0] = text
			last_time[0] = now
			# Wake-word guard (STT-based)
			if args.wake == "stt" and "jarvis" not in text.lower():
				return
			print(f"You: {text}")
			append_log("user", text)
			messages.append({"role": "user", "content": text})
			tool_msg = maybe_run_tool(text)
			if tool_msg:
				messages.append({"role": "assistant", "content": tool_msg})
				return
			if args.stream:
				full_text = ""
				partial = ""
				try:
					t_llm_start = time.time()
					for chunk in chat_ollama_stream(messages):
						full_text += chunk
						partial += chunk
						while "." in partial or "\n" in partial:
							sep = "." if "." in partial else "\n"
							sentence, partial = partial.split(sep, 1)
							sentence = sentence.strip()
							if sentence:
								if tts.is_speaking:
									tts.stop()
								tts.say(sentence)
					t_llm_end = time.time()
					_append_jsonl(METRICS_JSONL, {"ts": t_llm_end, "stage": "llm_stream", "duration_sec": t_llm_end - t_llm_start})
					if partial.strip():
						if tts.is_speaking:
							tts.stop()
						tts.say(partial.strip())
				finally:
					messages.append({"role": "assistant", "content": full_text.strip()})
					append_log("assistant", full_text.strip())
			else:
				t_llm_start = time.time()
				reply_bg = chat_ollama(messages)
				t_llm_end = time.time()
				_append_jsonl(METRICS_JSONL, {"ts": t_llm_end, "stage": "llm_blocking", "duration_sec": t_llm_end - t_llm_start})
				messages.append({"role": "assistant", "content": reply_bg})
				append_log("assistant", reply_bg)
				if tts.is_speaking:
					tts.stop()
				tts.say(reply_bg)

		stop = recognizer.listen_in_background(microphone, _bg_callback, phrase_time_limit=12.0)
		print("\nJarvis listening continuously. Say 'Jarvis' if --wake stt is enabled. Press Ctrl+C to stop.\n")
		try:
			console.print(Panel.fit("Jarvis listening continuously. Press Ctrl+C to stop.", title="Status", style="bold cyan"))
		except Exception:
			pass
		try:
			while True:
				time.sleep(0.1)
		except KeyboardInterrupt:
			stop()
			print("Goodbye.")
		return

	print("\nJarvis ready. Press Enter to talk, or type 'q' then Enter to quit.\n")
	try:
		console.print(Panel.fit("Jarvis ready. Press Enter to talk, or type 'q' then Enter to quit.", title="Status", style="bold cyan"))
	except Exception:
		pass
	while True:
		try:
			user_cmd = input("")
		except EOFError:
			print("No keyboard detected. Switching to --continuous mode.")
			# Simulate --continuous path
			bg_last_text: List[str] = [""]
			def _bg_callback(rec: sr.Recognizer, audio: sr.AudioData) -> None:
				try:
					if args.stt == "faster" and local_stt is not None:
						text = local_stt.transcribe_audio(audio, use_vad=(args.vad == "webrtc"))
					else:
						text = _recognize_google(rec, audio)
				except sr.UnknownValueError:
					return
				except Exception as exc:
					print(f"STT error: {exc}")
					return
				if not text:
					return
				if text.strip() == bg_last_text[0].strip():
					return
					bg_last_text[0] = text
					print(f"You: {text}")
				append_log("user", text)
				messages.append({"role": "user", "content": text})
				tool_msg = maybe_run_tool(text)
				if tool_msg:
					messages.append({"role": "assistant", "content": tool_msg})
					return
				if args.stream:
					full_text = ""
					partial = ""
					try:
						t_llm_start = time.time()
						for chunk in chat_ollama_stream(messages):
							full_text += chunk
							partial += chunk
							while "." in partial or "\n" in partial:
								sep = "." if "." in partial else "\n"
								sentence, partial = partial.split(sep, 1)
								sentence = sentence.strip()
								if sentence:
									if tts.is_speaking:
										tts.stop()
									tts.say(sentence)
							t_llm_end = time.time()
							_append_jsonl(METRICS_JSONL, {"ts": t_llm_end, "stage": "llm_stream", "duration_sec": t_llm_end - t_llm_start})
						if partial.strip():
							if tts.is_speaking:
								tts.stop()
							tts.say(partial.strip())
					finally:
						messages.append({"role": "assistant", "content": full_text.strip()})
						append_log("assistant", full_text.strip())
				else:
					t_llm_start = time.time()
					reply_bg = chat_ollama(messages)
					t_llm_end = time.time()
					_append_jsonl(METRICS_JSONL, {"ts": t_llm_end, "stage": "llm_blocking", "duration_sec": t_llm_end - t_llm_start})
					messages.append({"role": "assistant", "content": reply_bg})
					append_log("assistant", reply_bg)
					if tts.is_speaking:
						tts.stop()
					tts.say(reply_bg)

			stop = recognizer.listen_in_background(microphone, _bg_callback, phrase_time_limit=12.0)
			print("\nJarvis listening continuously. Press Ctrl+C to stop.\n")
			try:
				console.print(Panel.fit("Jarvis listening continuously. Press Ctrl+C to stop.", title="Status", style="bold cyan"))
			except Exception:
				pass
			try:
				while True:
					time.sleep(0.1)
			except KeyboardInterrupt:
				stop()
				print("Goodbye.")
			return
		if user_cmd.strip().lower() == "q":
			print("Goodbye.")
			break

		try:
			if args.stt == "faster" and local_stt is not None:
				with microphone as source:
					print("Listening... (speak now)")
					recognizer.adjust_for_ambient_noise(source, duration=0.4)  # type: ignore[arg-type]
					audio = recognizer.listen(source, timeout=6.0, phrase_time_limit=15.0)
					user_text = local_stt.transcribe_audio(audio, use_vad=(args.vad == "webrtc"))
			else:
				user_text = transcribe_from_mic(recognizer, microphone)
		except sr.WaitTimeoutError:
			print("Listening timed out. Press Enter and try again.")
			continue
		except Exception as exc:
			print(f"Listening error: {exc}")
			continue

		if not user_text:
			print("(Heard nothing intelligible.) Press Enter to try again.")
			continue

		print(f"You: {user_text}")
		append_log("user", user_text)
		messages.append({"role": "user", "content": user_text})
		tool_msg = maybe_run_tool(user_text)
		if tool_msg:
			messages.append({"role": "assistant", "content": tool_msg})
			continue

		if args.stream:
			full_text = ""
			partial = ""
			try:
				t_llm_start = time.time()
				for chunk in chat_ollama_stream(messages):
					full_text += chunk
					partial += chunk
					while "." in partial or "\n" in partial:
						sep = "." if "." in partial else "\n"
						sentence, partial = partial.split(sep, 1)
						sentence = sentence.strip()
						if sentence:
							if tts.is_speaking:
								tts.stop()
							tts.say(sentence)
				t_llm_end = time.time()
				_append_jsonl(METRICS_JSONL, {"ts": t_llm_end, "stage": "llm_stream", "duration_sec": t_llm_end - t_llm_start})
				if partial.strip():
					if tts.is_speaking:
						tts.stop()
					tts.say(partial.strip())
			finally:
				messages.append({"role": "assistant", "content": full_text.strip()})
				append_log("assistant", full_text.strip())
		else:
			t_llm_start = time.time()
			reply = chat_ollama(messages)
			t_llm_end = time.time()
			_append_jsonl(METRICS_JSONL, {"ts": t_llm_end, "stage": "llm_blocking", "duration_sec": t_llm_end - t_llm_start})
			append_log("assistant", reply)
			if not reply:
				reply = "I didn't get a response."
			messages.append({"role": "assistant", "content": reply})
			if tts.is_speaking:
				tts.stop()
			tts.say(reply)

		# Tiny pause to keep TTS responsive
		time.sleep(0.05)


if __name__ == "__main__":
	main()
