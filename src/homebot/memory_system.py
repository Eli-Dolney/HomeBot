"""
Memory System for Jarvis - Long-term memory with vector storage
Provides personalization and context retention across sessions
"""

import os
import json
import time
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False
    print("Warning: Vector dependencies not available. Install with: pip install sentence-transformers faiss-cpu")

@dataclass
class Memory:
    """Individual memory entry"""
    id: str
    content: str
    memory_type: str  # 'fact', 'preference', 'event', 'task'
    importance: float  # 0.0 to 1.0
    created_at: float
    last_accessed: float
    access_count: int
    tags: List[str]
    metadata: Dict[str, Any]

class MemorySystem:
    """Long-term memory system with vector storage and retrieval"""
    
    def __init__(self, memory_dir: str = "memory", max_memories: int = 1000):
        self.memory_dir = memory_dir
        self.max_memories = max_memories
        self.db_path = os.path.join(memory_dir, "memories.db")
        self.vector_index_path = os.path.join(memory_dir, "vector_index.faiss")
        self.vector_metadata_path = os.path.join(memory_dir, "vector_metadata.json")
        
        # Initialize directories
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize vector system if available
        self.encoder = None
        self.vector_index = None
        self.vector_metadata = []
        
        if HAS_VECTOR_DEPS:
            self._init_vector_system()
    
    def _init_database(self):
        """Initialize SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                tags TEXT,  -- JSON array
                metadata TEXT  -- JSON object
            )
        """)
        
        # Create indexes for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON memories(last_accessed)")
        
        conn.commit()
        conn.close()
    
    def _init_vector_system(self):
        """Initialize vector embedding system"""
        try:
            # Use a lightweight model for local embedding
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load or create vector index
            if os.path.exists(self.vector_index_path) and os.path.exists(self.vector_metadata_path):
                self.vector_index = faiss.read_index(self.vector_index_path)
                with open(self.vector_metadata_path, 'r') as f:
                    self.vector_metadata = json.load(f)
            else:
                # Create new index
                dimension = self.encoder.get_sentence_embedding_dimension()
                self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                self.vector_metadata = []
        except Exception as exc:
            print(f"Warning: Could not initialize vector system: {exc}")
            self.encoder = None
            self.vector_index = None
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _update_access_stats(self, memory_id: str):
        """Update access statistics for a memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (time.time(), memory_id))
        
        conn.commit()
        conn.close()
    
    def store_memory(self, content: str, memory_type: str = "fact", 
                    importance: float = 0.5, tags: List[str] = None, 
                    metadata: Dict[str, Any] = None) -> str:
        """Store a new memory"""
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
        
        memory_id = self._generate_id(content)
        now = time.time()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, content, memory_type, importance, created_at, last_accessed, 
             access_count, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, content, memory_type, importance, now, now, 0,
            json.dumps(tags), json.dumps(metadata)
        ))
        
        conn.commit()
        conn.close()
        
        # Add to vector index if available
        if self.encoder and self.vector_index is not None:
            try:
                embedding = self.encoder.encode([content])
                self.vector_index.add(embedding.astype('float32'))
                self.vector_metadata.append({
                    'id': memory_id,
                    'content': content,
                    'type': memory_type,
                    'importance': importance
                })
                
                # Save vector index
                faiss.write_index(self.vector_index, self.vector_index_path)
                with open(self.vector_metadata_path, 'w') as f:
                    json.dump(self.vector_metadata, f)
            except Exception as exc:
                print(f"Warning: Could not add to vector index: {exc}")
        
        return memory_id
    
    def retrieve_memories(self, query: str = None, memory_type: str = None, 
                         limit: int = 10, min_importance: float = 0.0) -> List[Memory]:
        """Retrieve memories based on query or filters"""
        memories = []
        
        if query and self.encoder and self.vector_index is not None:
            # Vector similarity search
            try:
                query_embedding = self.encoder.encode([query])
                scores, indices = self.vector_index.search(query_embedding.astype('float32'), limit * 2)
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.vector_metadata):
                        meta = self.vector_metadata[idx]
                        memory = self.get_memory_by_id(meta['id'])
                        if memory and memory.importance >= min_importance:
                            memories.append(memory)
                            if len(memories) >= limit:
                                break
            except Exception as exc:
                print(f"Warning: Vector search failed: {exc}")
        
        # Fallback to database search
        if not memories:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql = "SELECT * FROM memories WHERE importance >= ?"
            params = [min_importance]
            
            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type)
            
            sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            for row in rows:
                memory = Memory(
                    id=row[0],
                    content=row[1],
                    memory_type=row[2],
                    importance=row[3],
                    created_at=row[4],
                    last_accessed=row[5],
                    access_count=row[6],
                    tags=json.loads(row[7]) if row[7] else [],
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                memories.append(memory)
            
            conn.close()
        
        # Update access statistics
        for memory in memories:
            self._update_access_stats(memory.id)
        
        return memories
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Memory(
                id=row[0],
                content=row[1],
                memory_type=row[2],
                importance=row[3],
                created_at=row[4],
                last_accessed=row[5],
                access_count=row[6],
                tags=json.loads(row[7]) if row[7] else [],
                metadata=json.loads(row[8]) if row[8] else {}
            )
        return None
    
    def update_memory(self, memory_id: str, **updates) -> bool:
        """Update an existing memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['content', 'memory_type', 'importance', 'tags', 'metadata']:
                if key in ['tags', 'metadata']:
                    value = json.dumps(value)
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            conn.close()
            return False
        
        params.append(memory_id)
        sql = f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?"
        
        cursor.execute(sql, params)
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    def cleanup_old_memories(self, days_old: int = 30, min_importance: float = 0.3):
        """Remove old, low-importance memories"""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM memories 
            WHERE created_at < ? AND importance < ? AND access_count < 3
        """, (cutoff_time, min_importance))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]
        
        # By type
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
        by_type = dict(cursor.fetchall())
        
        # Average importance
        cursor.execute("SELECT AVG(importance) FROM memories")
        avg_importance = cursor.fetchone()[0] or 0.0
        
        # Recent activity
        week_ago = time.time() - (7 * 24 * 60 * 60)
        cursor.execute("SELECT COUNT(*) FROM memories WHERE last_accessed > ?", (week_ago,))
        recent_access = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_memories': total_memories,
            'by_type': by_type,
            'average_importance': avg_importance,
            'recent_access_count': recent_access,
            'vector_system_available': self.encoder is not None
        }

# Global memory system instance
_memory_system = None

def get_memory_system() -> MemorySystem:
    """Get the global memory system instance"""
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem()
    return _memory_system

def remember(content: str, memory_type: str = "fact", importance: float = 0.5, 
            tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
    """Convenience function to store a memory"""
    return get_memory_system().store_memory(content, memory_type, importance, tags, metadata)

def recall(query: str = None, memory_type: str = None, limit: int = 5) -> List[Memory]:
    """Convenience function to retrieve memories"""
    return get_memory_system().retrieve_memories(query, memory_type, limit)

def forget(memory_id: str) -> bool:
    """Convenience function to delete a memory"""
    return get_memory_system().delete_memory(memory_id)
