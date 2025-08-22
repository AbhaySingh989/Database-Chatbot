"""
Security Measures and Performance Optimizations for DataSuperAgent

This module provides security validation, SQL injection prevention,
file validation, caching mechanisms, and performance optimizations.
"""

import os
import re
import hashlib
import sqlite3
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache, wraps
import pandas as pd
from pathlib import Path
import streamlit as st


class SecurityValidator:
    """Security validation and protection mechanisms"""
    
    # Dangerous SQL patterns to block
    DANGEROUS_SQL_PATTERNS = [
        r'\bDROP\s+TABLE\b',
        r'\bDELETE\s+FROM\b',
        r'\bUPDATE\s+\w+\s+SET\b',
        r'\bINSERT\s+INTO\b',
        r'\bALTER\s+TABLE\b',
        r'\bCREATE\s+TABLE\b',
        r'\bTRUNCATE\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'\bUNION\s+ALL\b',
        r'--',
        r'/\*.*\*/',
        r'\bATTACH\s+DATABASE\b',
        r'\bDETACH\s+DATABASE\b',
        r'\bPRAGMA\b',
        r'\bVACUUM\b',
        r'\bREINDEX\b'
    ]
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'csv': 100 * 1024 * 1024,      # 100MB
        'xlsx': 50 * 1024 * 1024,      # 50MB
        'xls': 50 * 1024 * 1024,       # 50MB
        'db': 500 * 1024 * 1024,       # 500MB
        'sqlite': 500 * 1024 * 1024,   # 500MB
        'sqlite3': 500 * 1024 * 1024   # 500MB
    }
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        'csv', 'xlsx', 'xls', 'db', 'sqlite', 'sqlite3'
    }
    
    @classmethod
    def validate_sql_query(cls, query: str) -> Tuple[bool, str]:
        """Validate SQL query for security threats"""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        query_upper = query.upper().strip()
        
        # Must start with SELECT
        if not query_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_SQL_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                return False, f"Query contains dangerous operation: {pattern}"
        
        # Check for excessive complexity
        if len(query) > 10000:  # 10KB limit
            return False, "Query is too long (max 10KB)"
        
        # Check for excessive nested queries
        nested_count = query_upper.count('SELECT')
        if nested_count > 5:
            return False, "Too many nested SELECT statements (max 5)"
        
        return True, "Query is safe"
    
    @classmethod
    def validate_file_upload(cls, file_path: str, file_size: int) -> Tuple[bool, str]:
        """Validate uploaded file for security"""
        path = Path(file_path)
        
        # Check file extension
        extension = path.suffix.lower().lstrip('.')
        if extension not in cls.ALLOWED_EXTENSIONS:
            return False, f"File type '{extension}' not allowed"
        
        # Check file size
        max_size = cls.MAX_FILE_SIZES.get(extension, 10 * 1024 * 1024)  # Default 10MB
        if file_size > max_size:
            return False, f"File too large (max {max_size // (1024*1024)}MB for {extension})"
        
        # Check for suspicious file names
        suspicious_patterns = [r'\.\.', r'[<>:"|?*]', r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$']
        for pattern in suspicious_patterns:
            if re.search(pattern, path.name, re.IGNORECASE):
                return False, "Suspicious file name detected"
        
        return True, "File is safe"
    
    @classmethod
    def validate_database_file(cls, db_path: str) -> Tuple[bool, str]:
        """Validate SQLite database file"""
        try:
            # Basic file validation
            if not os.path.exists(db_path):
                return False, "Database file does not exist"
            
            file_size = os.path.getsize(db_path)
            is_valid, msg = cls.validate_file_upload(db_path, file_size)
            if not is_valid:
                return False, msg
            
            # Try to open as SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if it's a valid SQLite database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            
            # Check for suspicious content
            cursor.execute("SELECT COUNT(*) FROM sqlite_master")
            table_count = cursor.fetchone()[0]
            
            if table_count > 1000:  # Arbitrary limit
                conn.close()
                return False, "Database has too many tables (potential security risk)"
            
            conn.close()
            return True, "Database file is valid and safe"
            
        except sqlite3.DatabaseError as e:
            return False, f"Invalid SQLite database: {str(e)}"
        except Exception as e:
            return False, f"Database validation failed: {str(e)}"
    
    @classmethod
    def sanitize_column_names(cls, columns: List[str]) -> List[str]:
        """Sanitize column names to prevent injection"""
        sanitized = []
        for col in columns:
            # Remove dangerous characters and limit length
            safe_col = re.sub(r'[^\w\s-]', '', str(col))[:100]
            if safe_col and safe_col not in sanitized:
                sanitized.append(safe_col)
        return sanitized
    
    @classmethod
    def create_secure_temp_file(cls, suffix: str = '') -> str:
        """Create a secure temporary file"""
        temp_dir = tempfile.gettempdir()
        
        # Create secure temp file
        fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
        os.close(fd)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(temp_path, 0o600)
        
        return temp_path


class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self):
        self.query_cache = {}
        self.file_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'cache_size': 0
        }
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, dict):
            return hashlib.md5(str(sorted(data.items())).encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def cache_query_result(self, query: str, result: pd.DataFrame, ttl: int = 300):
        """Cache query result with TTL (time to live)"""
        cache_key = self.get_cache_key(query)
        
        # Limit cache size
        if len(self.query_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k]['timestamp'])
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = {
            'result': result.copy(),
            'timestamp': time.time(),
            'ttl': ttl
        }
        
        self.cache_stats['cache_size'] = len(self.query_cache)
    
    def get_cached_query_result(self, query: str) -> Optional[pd.DataFrame]:
        """Get cached query result if available and valid"""
        cache_key = self.get_cache_key(query)
        
        if cache_key in self.query_cache:
            cached_item = self.query_cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - cached_item['timestamp'] < cached_item['ttl']:
                self.cache_stats['hits'] += 1
                return cached_item['result'].copy()
            else:
                # Remove expired cache
                del self.query_cache[cache_key]
                self.cache_stats['cache_size'] = len(self.query_cache)
        
        self.cache_stats['misses'] += 1
        return None
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if it has few unique values
                unique_ratio = len(optimized_df[col].unique()) / len(optimized_df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
            
            elif col_type in ['int64', 'int32']:
                # Downcast integers
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            
            elif col_type in ['float64', 'float32']:
                # Downcast floats
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': self.cache_stats['cache_size'],
            'memory_usage_mb': self._estimate_cache_memory()
        }
    
    def _estimate_cache_memory(self) -> float:
        """Estimate cache memory usage in MB"""
        total_size = 0
        for cached_item in self.query_cache.values():
            df = cached_item['result']
            total_size += df.memory_usage(deep=True).sum()
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def clear_cache(self):
        """Clear all caches"""
        self.query_cache.clear()
        self.file_cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0, 'cache_size': 0}


class ConnectionManager:
    """Manage database connections with pooling and cleanup"""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.active_connections = {}
        self.connection_stats = {
            'total_created': 0,
            'total_closed': 0,
            'active_count': 0
        }
    
    def get_connection(self, db_path: str) -> sqlite3.Connection:
        """Get database connection with connection pooling"""
        connection_key = os.path.abspath(db_path)
        
        # Check if connection already exists and is valid
        if connection_key in self.active_connections:
            conn = self.active_connections[connection_key]
            try:
                # Test connection
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                # Connection is invalid, remove it
                del self.active_connections[connection_key]
                self.connection_stats['active_count'] -= 1
        
        # Create new connection if under limit
        if len(self.active_connections) >= self.max_connections:
            # Close oldest connection
            oldest_key = next(iter(self.active_connections))
            self.close_connection(oldest_key)
        
        # Create new connection
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Set performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        self.active_connections[connection_key] = conn
        self.connection_stats['total_created'] += 1
        self.connection_stats['active_count'] += 1
        
        return conn
    
    def close_connection(self, db_path: str):
        """Close specific database connection"""
        connection_key = os.path.abspath(db_path)
        
        if connection_key in self.active_connections:
            try:
                self.active_connections[connection_key].close()
            except:
                pass
            
            del self.active_connections[connection_key]
            self.connection_stats['total_closed'] += 1
            self.connection_stats['active_count'] -= 1
    
    def close_all_connections(self):
        """Close all database connections"""
        for connection_key in list(self.active_connections.keys()):
            self.close_connection(connection_key)
    
    def get_connection_stats(self) -> Dict[str, int]:
        """Get connection statistics"""
        return self.connection_stats.copy()


# Global instances
security_validator = SecurityValidator()
performance_optimizer = PerformanceOptimizer()
connection_manager = ConnectionManager()


def secure_query_execution(func):
    """Decorator for secure query execution with caching"""
    @wraps(func)
    def wrapper(self, query: str, *args, **kwargs):
        # Security validation
        is_safe, error_msg = security_validator.validate_sql_query(query)
        if not is_safe:
            raise ValueError(f"Security validation failed: {error_msg}")
        
        # Execute the original method
        result = func(self, query, *args, **kwargs)
        
        return result
    
    return wrapper


def monitor_performance():
    """Display performance monitoring information"""
    if st.sidebar.checkbox("Show Performance Stats", key="perf_stats_checkbox"):
        with st.sidebar.expander("📊 Performance Monitor", expanded=False):
            cache_stats = performance_optimizer.get_cache_statistics()
            conn_stats = connection_manager.get_connection_stats()
            
            st.write("**Cache Performance:**")
            st.write(f"• Hit Rate: {cache_stats['hit_rate_percent']}%")
            st.write(f"• Cache Size: {cache_stats['cache_size']} items")
            st.write(f"• Memory Usage: {cache_stats['memory_usage_mb']:.1f} MB")
            
            st.write("**Database Connections:**")
            st.write(f"• Active: {conn_stats['active_count']}")
            st.write(f"• Total Created: {conn_stats['total_created']}")
            st.write(f"• Total Closed: {conn_stats['total_closed']}")
            
            if st.button("Clear Cache", key="clear_cache_btn"):
                performance_optimizer.clear_cache()
                st.success("Cache cleared!")
                st.rerun()


def cleanup_resources():
    """Clean up all resources on app shutdown"""
    connection_manager.close_all_connections()
    performance_optimizer.clear_cache()


# Register cleanup function
import atexit
atexit.register(cleanup_resources)