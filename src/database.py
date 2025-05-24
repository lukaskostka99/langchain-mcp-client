"""
Database and storage management for the LangChain MCP Client.

This module handles persistent conversation storage using SQLite
and provides interfaces for managing conversation metadata.
"""

import sqlite3
import threading
import datetime
from pathlib import Path
from typing import Dict, List
from contextlib import ExitStack

import streamlit as st
import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class PersistentStorageManager:
    """Manages SQLite database for persistent conversation storage."""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create conversations metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_metadata (
                        thread_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        message_count INTEGER DEFAULT 0,
                        last_message TEXT
                    )
                """)
                
                conn.commit()
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")
    
    def get_checkpointer(self):
        """Get a SQLite checkpointer using context manager pattern."""
        try:
            # Use ExitStack to manage the context manager manually
            if not hasattr(self, '_stack'):
                self._stack = ExitStack()
            
            # Enter the context manager and keep it open
            checkpointer = self._stack.enter_context(
                AsyncSqliteSaver.from_conn_string(str(self.db_path))
            )
            
            return checkpointer
        except Exception as e:
            st.error(f"Error creating SQLite checkpointer: {str(e)}")
            # Fallback to in-memory if SQLite fails
            return InMemorySaver()
    
    def close_checkpointer(self):
        """Close the checkpointer context."""
        try:
            if hasattr(self, '_stack'):
                self._stack.close()
                delattr(self, '_stack')
        except Exception as e:
            st.warning(f"Error closing checkpointer: {str(e)}")
    
    async def get_async_checkpointer(self):
        """Get an async SQLite checkpointer."""
        try:
            # Create an async SQLite connection
            conn = await aiosqlite.connect(str(self.db_path))
            checkpointer = AsyncSqliteSaver(conn)
            return checkpointer
        except Exception as e:
            st.error(f"Error creating async SQLite checkpointer: {str(e)}")
            # Fallback to in-memory if SQLite fails
            return InMemorySaver()
    
    def get_checkpointer_context_manager(self):
        """Get a SQLite checkpointer context manager for the given thread."""
        return AsyncSqliteSaver.from_conn_string(str(self.db_path))
    
    def list_conversations(self) -> List[Dict]:
        """List all stored conversations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT thread_id, created_at, updated_at, title, message_count, last_message
                    FROM conversation_metadata
                    ORDER BY updated_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error listing conversations: {str(e)}")
            return []
    
    def update_conversation_metadata(self, thread_id: str, title: str = None, 
                                   message_count: int = None, last_message: str = None):
        """Update conversation metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or update conversation metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO conversation_metadata 
                    (thread_id, created_at, updated_at, title, message_count, last_message)
                    VALUES (?, 
                            COALESCE((SELECT created_at FROM conversation_metadata WHERE thread_id = ?), CURRENT_TIMESTAMP),
                            CURRENT_TIMESTAMP, 
                            COALESCE(?, (SELECT title FROM conversation_metadata WHERE thread_id = ?)),
                            COALESCE(?, (SELECT message_count FROM conversation_metadata WHERE thread_id = ?)),
                            COALESCE(?, (SELECT last_message FROM conversation_metadata WHERE thread_id = ?)))
                """, (thread_id, thread_id, title, thread_id, message_count, thread_id, last_message, thread_id))
                
                conn.commit()
        except Exception as e:
            st.error(f"Error updating conversation metadata: {str(e)}")
    
    def delete_conversation(self, thread_id: str):
        """Delete a conversation and its metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete metadata
                cursor.execute("DELETE FROM conversation_metadata WHERE thread_id = ?", (thread_id,))
                
                # Note: LangGraph's SQLite checkpointer handles its own tables
                # We only delete our metadata here
                
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error deleting conversation: {str(e)}")
            return False
    
    def export_conversation(self, thread_id: str) -> Dict:
        """Export a conversation's data."""
        try:
            # Get metadata
            metadata = None
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM conversation_metadata WHERE thread_id = ?", (thread_id,))
                row = cursor.fetchone()
                if row:
                    metadata = dict(row)
            
            # Get checkpoints (this would require access to LangGraph's internal tables)
            # For now, we'll return the metadata and note that checkpoint data is handled by LangGraph
            return {
                'thread_id': thread_id,
                'metadata': metadata,
                'export_timestamp': datetime.datetime.now().isoformat(),
                'note': 'Checkpoint data is managed by LangGraph SQLite checkpointer'
            }
        except Exception as e:
            st.error(f"Error exporting conversation: {str(e)}")
            return {}
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get conversation count
                cursor.execute("SELECT COUNT(*) FROM conversation_metadata")
                conversation_count = cursor.fetchone()[0]
                
                # Get total messages
                cursor.execute("SELECT SUM(message_count) FROM conversation_metadata")
                total_messages = cursor.fetchone()[0] or 0
                
                # Get database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'conversation_count': conversation_count,
                    'total_messages': total_messages,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'database_path': str(self.db_path)
                }
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
            return {} 