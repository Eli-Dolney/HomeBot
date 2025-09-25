"""
Security and Privacy Configuration for Jarvis
Ensures offline-first operation and data protection
"""

import os
import socket
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    offline_mode: bool = True
    allowed_hosts: List[str] = None
    blocked_domains: List[str] = None
    data_retention_days: int = 30
    encrypt_memories: bool = False
    require_confirmation: bool = True
    log_sensitive_data: bool = False
    
    def __post_init__(self):
        if self.allowed_hosts is None:
            self.allowed_hosts = ["localhost", "127.0.0.1"]
        if self.blocked_domains is None:
            self.blocked_domains = [
                "google.com", "openai.com", "anthropic.com", 
                "microsoft.com", "amazon.com", "facebook.com"
            ]

class SecurityManager:
    """Manages security and privacy settings"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.network_monitor = NetworkMonitor()
        self.data_cleaner = DataCleaner(self.config.data_retention_days)
    
    def check_offline_mode(self) -> bool:
        """Verify that the system is operating in offline mode"""
        if not self.config.offline_mode:
            return True
        
        # Check for any outbound network connections
        active_connections = self.network_monitor.get_active_connections()
        for conn in active_connections:
            if not self._is_allowed_connection(conn):
                return False
        
        return True
    
    def _is_allowed_connection(self, connection: Dict[str, Any]) -> bool:
        """Check if a connection is allowed"""
        remote_host = connection.get('remote_host', '')
        remote_port = connection.get('remote_port', 0)
        
        # Allow local connections
        if remote_host in self.config.allowed_hosts:
            return True
        
        # Allow Ollama on default port
        if remote_port == 11434 and remote_host in ['localhost', '127.0.0.1']:
            return True
        
        # Block everything else in offline mode
        return False
    
    def validate_tool_execution(self, command: str, args: List[str]) -> bool:
        """Validate if a tool command is safe to execute"""
        # Block network-related commands
        network_commands = ['curl', 'wget', 'ping', 'nslookup', 'dig', 'ssh', 'scp', 'rsync']
        if command in network_commands:
            return False
        
        # Block potentially dangerous commands
        dangerous_commands = ['rm', 'del', 'format', 'fdisk', 'mkfs', 'dd']
        if command in dangerous_commands:
            return False
        
        # Block commands that could access external resources
        for arg in args:
            if any(domain in arg for domain in self.config.blocked_domains):
                return False
        
        return True
    
    def sanitize_log_data(self, data: str) -> str:
        """Remove sensitive information from log data"""
        if self.config.log_sensitive_data:
            return data
        
        # Remove potential sensitive patterns
        import re
        patterns = [
            r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+',  # passwords
            r'token["\']?\s*[:=]\s*["\']?[^"\'\s]+',     # tokens
            r'key["\']?\s*[:=]\s*["\']?[^"\'\s]+',       # keys
            r'secret["\']?\s*[:=]\s*["\']?[^"\'\s]+',    # secrets
        ]
        
        for pattern in patterns:
            data = re.sub(pattern, r'\1=***REDACTED***', data, flags=re.IGNORECASE)
        
        return data
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """Clean up old data based on retention policy"""
        return self.data_cleaner.cleanup()

class NetworkMonitor:
    """Monitor network connections to ensure offline operation"""
    
    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active network connections"""
        connections = []
        
        try:
            # Use netstat to get active connections
            result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'ESTABLISHED' in line or 'LISTEN' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        local_addr = parts[3]
                        remote_addr = parts[4] if len(parts) > 4 else ''
                        
                        if remote_addr and remote_addr != '*:*':
                            # Parse remote address
                            if ':' in remote_addr:
                                host, port = remote_addr.rsplit(':', 1)
                                connections.append({
                                    'remote_host': host,
                                    'remote_port': int(port),
                                    'local_addr': local_addr
                                })
        except Exception:
            pass
        
        return connections
    
    def is_offline(self) -> bool:
        """Check if system is truly offline"""
        try:
            # Try to resolve a known external domain
            socket.gethostbyname('google.com')
            return False  # We can reach external sites
        except socket.gaierror:
            return True  # Truly offline

class DataCleaner:
    """Clean up old data based on retention policies"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
    
    def cleanup(self) -> Dict[str, int]:
        """Clean up old data files"""
        cleaned = {
            'logs': 0,
            'sessions': 0,
            'memories': 0,
            'temp_files': 0
        }
        
        import time
        cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
        
        # Clean up old log files
        cleaned['logs'] = self._cleanup_directory('logs', cutoff_time)
        
        # Clean up old session files
        cleaned['sessions'] = self._cleanup_directory('sessions', cutoff_time)
        
        # Clean up old memory files
        cleaned['memories'] = self._cleanup_directory('memory', cutoff_time)
        
        # Clean up temporary files
        cleaned['temp_files'] = self._cleanup_temp_files(cutoff_time)
        
        return cleaned
    
    def _cleanup_directory(self, directory: str, cutoff_time: float) -> int:
        """Clean up files in a directory older than cutoff time"""
        if not os.path.exists(directory):
            return 0
        
        cleaned_count = 0
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1
        except Exception:
            pass
        
        return cleaned_count
    
    def _cleanup_temp_files(self, cutoff_time: float) -> int:
        """Clean up temporary files"""
        import tempfile
        temp_dir = tempfile.gettempdir()
        cleaned_count = 0
        
        try:
            for file in os.listdir(temp_dir):
                if file.startswith('jarvis_') or file.startswith('tmp_jarvis_'):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1
        except Exception:
            pass
        
        return cleaned_count

# Global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get the global security manager instance"""
    global _security_manager
    if _security_manager is None:
        config = SecurityConfig()
        _security_manager = SecurityManager(config)
    return _security_manager

def check_security() -> Dict[str, Any]:
    """Perform security checks and return status"""
    manager = get_security_manager()
    
    return {
        'offline_mode': manager.check_offline_mode(),
        'network_monitor': manager.network_monitor.is_offline(),
        'data_retention_days': manager.config.data_retention_days,
        'encrypt_memories': manager.config.encrypt_memories,
        'require_confirmation': manager.config.require_confirmation
    }

def cleanup_data() -> Dict[str, int]:
    """Clean up old data"""
    manager = get_security_manager()
    return manager.cleanup_old_data()
