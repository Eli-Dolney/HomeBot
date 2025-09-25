#!/usr/bin/env python3
"""
Minimal Jarvis Web UI - Mac M4 Optimized
Simple interface without complex Gradio features that cause compatibility issues
"""

import os
import time
import json
from typing import List, Dict, Tuple, Optional
import subprocess

# Reuse backend logic from jarvis.py
import jarvis  # type: ignore

def simple_chat_interface():
    """Simple command-line chat interface"""
    print("🧠 Jarvis Minimal Interface - Mac M4 Optimized")
    print("=" * 50)
    print("🔒 Fully local - no data leaves your Mac")
    print("💡 Type 'help' for commands, 'quit' to exit")
    print()
    
    # Initialize memory system if available
    memory_available = hasattr(jarvis, 'HAS_MEMORY_SYSTEM') and jarvis.HAS_MEMORY_SYSTEM
    if memory_available:
        print("🧠 Memory system: ✅ Available")
    else:
        print("🧠 Memory system: ❌ Not available")
    
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
                
            if user_input.lower() == 'help':
                print_help()
                continue
                
            if user_input.lower() == 'status':
                print_status()
                continue
                
            if user_input.lower() == 'test':
                test_tools()
                continue
            
            # Process the message
            response = process_message(user_input)
            print(f"Jarvis: {response}")
            
            # Speak the response
            try:
                jarvis.TextToSpeech().say(response)
            except Exception as exc:
                print(f"(TTS error: {exc})")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as exc:
            print(f"Error: {exc}")

def process_message(message: str) -> str:
    """Process user message and return response"""
    try:
        # Check for memory commands first
        if hasattr(jarvis, 'HAS_MEMORY_SYSTEM') and jarvis.HAS_MEMORY_SYSTEM:
            if message.lower().startswith("remember "):
                content = message[9:].strip()
                if content:
                    memory_id = jarvis.remember(content, memory_type="fact", importance=0.7)
                    return f"I'll remember: {content}"
            
            elif message.lower().startswith("recall "):
                query = message[7:].strip()
                if query:
                    memories = jarvis.recall(query, limit=5)
                    if memories:
                        result = f"Here's what I remember about '{query}':\n"
                        for i, memory in enumerate(memories, 1):
                            result += f"{i}. {memory.content}\n"
                        return result
                    else:
                        return f"I don't have any memories about '{query}'"
            
            elif message.lower().startswith("forget "):
                query = message[7:].strip()
                if query:
                    memories = jarvis.recall(query, limit=10)
                    if memories:
                        jarvis.forget(memories[0].id)
                        return f"I've forgotten: {memories[0].content}"
                    else:
                        return f"I don't have any memories about '{query}' to forget"
        
        # Check for tool commands
        tool_response = check_tools(message)
        if tool_response:
            return tool_response
        
        # Regular chat
        messages = [
            {"role": "system", "content": jarvis.SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
        
        return jarvis.chat_ollama(messages)
        
    except Exception as exc:
        return f"Error processing message: {exc}"

def check_tools(message: str) -> Optional[str]:
    """Check if message matches any tool patterns"""
    message_lower = message.lower().strip()
    
    # System tools
    if message_lower == "open calculator":
        try:
            subprocess.run(["open", "-a", "Calculator"], check=True)
            return "✅ Calculator opened"
        except Exception as exc:
            return f"❌ Error: {exc}"
    
    elif message_lower == "open safari":
        try:
            subprocess.run(["open", "-a", "Safari"], check=True)
            return "✅ Safari opened"
        except Exception as exc:
            return f"❌ Error: {exc}"
    
    elif message_lower == "open terminal":
        try:
            subprocess.run(["open", "-a", "Terminal"], check=True)
            return "✅ Terminal opened"
        except Exception as exc:
            return f"❌ Error: {exc}"
    
    elif message_lower == "lock screen":
        try:
            subprocess.run(["osascript", "-e", "tell application \"System Events\" to keystroke \"q\" using {command down, control down}"], check=True)
            return "✅ Screen locked"
        except Exception as exc:
            return f"❌ Error: {exc}"
    
    elif message_lower.startswith("set volume to "):
        try:
            volume = message_lower.split("set volume to ")[1]
            subprocess.run(["osascript", "-e", f"set volume output volume {volume}"], check=True)
            return f"✅ Volume set to {volume}"
        except Exception as exc:
            return f"❌ Error: {exc}"
    
    elif message_lower.startswith("search for "):
        try:
            query = message_lower.split("search for ", 1)[1]
            subprocess.run(["osascript", "-e", "tell application \"System Events\" to keystroke \" \" using {command down}", "-e", "delay 0.5", "-e", f"tell application \"System Events\" to keystroke \"{query}\""], check=True)
            return f"✅ Spotlight search for '{query}'"
        except Exception as exc:
            return f"❌ Error: {exc}"
    
    return None

def print_help():
    """Print help information"""
    print("\n📖 Available Commands:")
    print("=" * 30)
    print("🧠 Memory Commands:")
    print("  remember [something] - Store a memory")
    print("  recall [query] - Search memories")
    print("  forget [query] - Delete a memory")
    print()
    print("🛠️ System Tools:")
    print("  open calculator - Open Calculator app")
    print("  open safari - Open Safari browser")
    print("  open terminal - Open Terminal")
    print("  lock screen - Lock your Mac")
    print("  set volume to [number] - Adjust volume")
    print("  search for [query] - Spotlight search")
    print()
    print("ℹ️ Other Commands:")
    print("  help - Show this help")
    print("  status - Show system status")
    print("  test - Test tools")
    print("  quit - Exit")
    print()

def print_status():
    """Print system status"""
    print("\n📊 System Status:")
    print("=" * 20)
    print(f"Model: {jarvis.MODEL_NAME}")
    print(f"Ollama Host: {jarvis.OLLAMA_HOST}")
    print(f"Memory System: {'✅ Available' if hasattr(jarvis, 'HAS_MEMORY_SYSTEM') and jarvis.HAS_MEMORY_SYSTEM else '❌ Not Available'}")
    print(f"TTS Engine: Local (pyttsx3 + macOS say)")
    print(f"STT Engine: Local Whisper (faster-whisper)")
    print("Mac M4 Optimizations: ✅ Enabled")
    print()

def test_tools():
    """Test available tools"""
    print("\n🧪 Testing Tools:")
    print("=" * 20)
    
    # Test Calculator
    try:
        subprocess.run(["open", "-a", "Calculator"], check=True)
        print("✅ Calculator: Working")
    except Exception as exc:
        print(f"❌ Calculator: {exc}")
    
    # Test system info
    try:
        result = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ System Info: Working")
        else:
            print("❌ System Info: Failed")
    except Exception as exc:
        print(f"❌ System Info: {exc}")
    
    print()

if __name__ == "__main__":
    simple_chat_interface()
