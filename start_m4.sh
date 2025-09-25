#!/bin/bash

# Mac M4 Optimized Jarvis Startup Script
# This script optimizes Jarvis for Mac M4 with the best settings

echo "🚀 Starting Jarvis on Mac M4..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running. Please start it first:"
    echo "   ollama serve"
    exit 1
fi

# Check if the model is available
MODEL="llama3.2:3b-instruct-q4_K_M"
if ! ollama list | grep -q "$MODEL"; then
    echo "📥 Downloading optimized model for Mac M4..."
    ollama pull "$MODEL"
fi

# Set optimal environment variables for Mac M4
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="$MODEL"
export OLLAMA_TEMPERATURE="0.7"
export TTS_RATE_WPM="185"
export TTS_VOICE=""  # Use system default
export TTS_LANGUAGE="en"

# Mac M4 specific optimizations
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "✅ Environment configured for Mac M4"
echo "🎤 Starting Jarvis with continuous listening..."
echo "💡 Say 'Jarvis' to activate, or just start talking"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Start Jarvis with Mac M4 optimized settings
python3 jarvis.py \
    --continuous \
    --stream \
    --stt faster \
    --vad webrtc \
    --whisper-model small \
    --stt-compute auto \
    --device-name "MacBook" \
    --enable-tools \
    --tools-config "tools_m4.yaml" \
    --wake off \
    --profile default
