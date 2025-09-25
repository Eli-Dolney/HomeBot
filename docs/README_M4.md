# 🧠 Jarvis Enhanced - Mac M4 Optimized

**Privacy-first local AI assistant** with streaming responses, real-time voice input, Mac-specific tools, and long-term memory.

## ✨ What's New for Mac M4

### 🚀 **Performance Optimizations**
- **MPS Acceleration**: Uses Metal Performance Shaders for Mac M4
- **Optimized Compute Types**: Auto-detects best settings for Apple Silicon
- **Local STT**: faster-whisper with `int8_float16` for optimal performance
- **Streaming Responses**: Real-time token streaming for instant feedback

### 🛠️ **Enhanced Mac Tools**
- **System Control**: Volume, screen lock, sleep display
- **Spotlight Integration**: Voice-activated search
- **App Launcher**: Calculator, Safari, Terminal, VS Code, Xcode
- **Calendar & Notes**: Create events and notes via voice
- **Media Control**: Play/pause/skip music tracks

### 🧠 **Long-term Memory**
- **Vector Storage**: Semantic search through your conversations
- **Memory Commands**: `remember`, `recall`, `forget`
- **Context Awareness**: Automatically retrieves relevant memories
- **Privacy-First**: All data stays on your Mac

### 🔒 **Security & Privacy**
- **Offline-First**: No external network calls except Ollama
- **Data Retention**: Automatic cleanup of old data
- **Tool Confirmation**: Asks before executing commands
- **Encrypted Storage**: Optional memory encryption

## 🚀 Quick Start

### 1. **Automated Setup** (Recommended)
```bash
./setup_m4.sh
```

### 2. **Manual Setup**
```bash
# Install dependencies
brew install portaudio python@3.11
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_m4.txt

# Install Ollama and model
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.2:3b-instruct-q4_K_M
```

### 3. **Start Jarvis**
```bash
# Voice mode with Mac M4 optimizations
./start_m4.sh

# Enhanced web UI
python3 -m homebot.ui_enhanced
```

## 🎤 Voice Commands

### **Memory System**
- `"Remember I prefer dark mode"` - Store a memory
- `"Recall my preferences"` - Search memories
- `"Forget about that meeting"` - Delete a memory

### **System Control**
- `"Open Calculator"` - Launch Calculator app
- `"Set volume to 50"` - Adjust system volume
- `"Lock screen"` - Lock your Mac
- `"Search for Python tutorial"` - Spotlight search

### **Apps & Files**
- `"Open Safari"` - Launch Safari
- `"Open Terminal"` - Launch Terminal
- `"Open Code"` - Launch VS Code
- `"Show desktop"` - Show desktop

### **Calendar & Notes**
- `"Create calendar event Team meeting tomorrow"` - Add to Calendar
- `"Create note Buy groceries"` - Add to Notes
- `"Open Calendar"` - Launch Calendar app

### **Media Control**
- `"Play music"` - Start Music playback
- `"Pause music"` - Pause Music
- `"Next track"` - Skip to next song

## 🖥️ Web Interface

The enhanced web UI (`ui_enhanced.py`) provides:

- **💬 Enhanced Chat**: Streaming responses with better UX
- **🎤 Voice Input**: Record or upload audio for transcription
- **🛠️ Mac Tools**: Test and manage Mac-specific tools
- **⚙️ Settings**: Configure Jarvis and view system info

Access at: `http://127.0.0.1:7860`

## ⚙️ Configuration

### **Environment Variables**
```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="llama3.2:3b-instruct-q4_K_M"
export TTS_RATE_WPM="185"
export TTS_VOICE=""  # Use system default
```

### **Configuration Files**
- `config_m4.yaml` - Mac M4 optimized settings
- `configs/tools_m4.yaml` - Enhanced Mac-specific tools
- `security_config.py` - Privacy and security settings

## 🔧 Advanced Usage

### **Command Line Options**
```bash
# Continuous listening with Mac M4 optimizations
python3 jarvis.py --continuous --stream --stt faster --vad webrtc --stt-compute auto

# Text-only mode
python3 jarvis.py --text "What's the weather like?"

# Coding assistant profile
python3 jarvis.py --profile coding --continuous

# With tools enabled
python3 jarvis.py --enable-tools --tools-config configs/tools_m4.yaml
```

### **Memory Management**
```python
from memory_system import remember, recall, forget

# Store a memory
memory_id = remember("User prefers Python over JavaScript", importance=0.8)

# Search memories
memories = recall("programming preferences", limit=5)

# Delete a memory
forget(memory_id)
```

## 📊 Performance

### **Mac M4 Optimizations**
- **STT**: ~200ms latency with faster-whisper
- **LLM**: ~500ms first token with Ollama
- **TTS**: ~100ms with local synthesis
- **Memory**: ~50ms vector search

### **Resource Usage**
- **RAM**: ~2GB for full system
- **Storage**: ~5GB for models and data
- **CPU**: Optimized for M4 efficiency cores
- **GPU**: Uses MPS for acceleration

## 🔒 Privacy & Security

### **Data Protection**
- ✅ All processing happens locally
- ✅ No data sent to external services
- ✅ Automatic data cleanup
- ✅ Optional memory encryption
- ✅ Tool execution confirmation

### **Network Security**
- ✅ Offline-first operation
- ✅ Only allows localhost connections
- ✅ Blocks external domains
- ✅ Monitors network activity

## 🐛 Troubleshooting

### **Common Issues**

**TTS not working:**
```bash
# Test macOS say command
say "Hello world"

# Check voice settings
say -v ? | grep -i "alex\|samantha"
```

**Microphone not detected:**
```bash
# List available microphones
python3 -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"

# Grant microphone permission to Terminal
# System Preferences > Security & Privacy > Microphone
```

**Ollama connection issues:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve &
```

**Memory system not working:**
```bash
# Install missing dependencies
pip install sentence-transformers faiss-cpu

# Check memory directory
ls -la memory/
```

### **Performance Issues**

**Slow STT:**
- Use smaller Whisper model: `--whisper-model tiny`
- Disable VAD: `--vad off`
- Use Google STT: `--stt google`

**High memory usage:**
- Reduce model size: `--whisper-model tiny`
- Disable memory system: Remove memory dependencies
- Clean old data: Run cleanup script

## 📈 Roadmap

### **v0.3 (Next)**
- [ ] Real-time audio streaming in web UI
- [ ] Wake word detection with Porcupine
- [ ] Plugin system for custom tools
- [ ] Advanced memory management

### **v0.4 (Future)**
- [ ] Multi-user support
- [ ] Advanced TTS with voice cloning
- [ ] Integration with HomeKit
- [ ] Mobile companion app

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Mac M4
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM serving
- **faster-whisper** for local STT
- **sentence-transformers** for embeddings
- **faiss** for vector search
- **Gradio** for web interface

---

**Made with ❤️ for Mac M4 users who value privacy and performance.**
