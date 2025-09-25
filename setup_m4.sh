#!/bin/bash

# Mac M4 Setup Script for Jarvis
# This script sets up Jarvis with all optimizations for Mac M4

set -e  # Exit on any error

echo "🚀 Setting up Jarvis for Mac M4..."
echo "=================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This script is optimized for Apple Silicon (M1/M2/M3/M4)"
    echo "   It may still work on Intel Macs but won't be fully optimized"
fi

echo "✅ Detected macOS on $(uname -m) architecture"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew install portaudio python@3.11

# Check for Python 3.11+
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION < 3.11" | bc -l) -eq 1 ]]; then
    echo "⚠️  Python $PYTHON_VERSION detected. Python 3.11+ recommended for best performance"
fi

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo "🔥 Installing PyTorch with MPS support for Mac M4..."
pip install torch torchvision torchaudio

# Install core requirements
echo "📦 Installing core requirements..."
pip install -r requirements_m4.txt

# Install optional memory system dependencies
echo "🧠 Installing memory system dependencies..."
pip install sentence-transformers faiss-cpu || echo "⚠️  Memory system dependencies failed to install (optional)"

# Check for Ollama
echo "🤖 Checking for Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service
echo "🚀 Starting Ollama service..."
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve &
    sleep 3
    echo "✅ Ollama service started"
else
    echo "✅ Ollama service already running"
fi

# Download recommended model
echo "📥 Downloading optimized model for Mac M4..."
MODEL="llama3.2:3b-instruct-q4_K_M"
if ! ollama list | grep -q "$MODEL"; then
    ollama pull "$MODEL"
    echo "✅ Model downloaded"
else
    echo "✅ Model already available"
fi

# Set up directories
echo "📁 Setting up directories..."
mkdir -p logs sessions memory
echo "✅ Directories created"

# Set permissions
echo "🔐 Setting up permissions..."
chmod +x start_m4.sh
chmod +x setup_m4.sh
echo "✅ Permissions set"

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import jarvis
import memory_system
import security_config
print('✅ All modules imported successfully')
"

# Check MPS availability
echo "⚡ Checking MPS (Metal Performance Shaders) availability..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS is available - Mac M4 optimization enabled')
else:
    print('⚠️  MPS not available - using CPU fallback')
"

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo ""
echo "🚀 To start Jarvis with Mac M4 optimizations:"
echo "   ./start_m4.sh"
echo ""
echo "🌐 To start the enhanced web UI:"
echo "   python3 ui_enhanced.py"
echo ""
echo "🔧 Available commands:"
echo "   - 'remember [something]' - Store a memory"
echo "   - 'recall [query]' - Search memories"
echo "   - 'forget [query]' - Delete a memory"
echo "   - 'open calculator' - Open Calculator app"
echo "   - 'search for [query]' - Spotlight search"
echo "   - 'set volume to 50' - Adjust volume"
echo ""
echo "🔒 Security features:"
echo "   - Offline-first operation"
echo "   - Local data only"
echo "   - Automatic data cleanup"
echo "   - Tool execution confirmation"
echo ""
echo "📊 Performance optimizations:"
echo "   - MPS acceleration for Mac M4"
echo "   - Optimized compute types"
echo "   - Local STT with faster-whisper"
echo "   - Streaming responses"
echo ""
echo "Happy chatting with Jarvis! 🤖"
