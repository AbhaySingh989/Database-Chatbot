"""
SQLite Database Handler for DataSuperAgent

This module provides SQLite database connection management, schema exploration,
and query execution capabilities for the DataSuperAgent application.
"""

import sqlite3
import os
import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
from security_performance import (
    security_validator, performance_optimizer, connection_manager,
    secure_query_execution
)


@dataclass
class ColumnInfo:
    """Data model for database column information"""
    name: str
    data_type: str
    nullable: bool
    primary_key: bool
    default_value: Any = None


@dataclass
class TableInfo:
    """Data model for database table information"""
    name: str
    row_count: int
    columns: List[ColumnInfo]
    foreign_keys: List[Dict] = None
    indexes: List[str] = None

    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = []
        if self.indexes is None:
            self.indexes = []


@dataclass
class QueryResult:
    """Data model for query execution results"""
    dataframe: pd.DataFrame
    query: str
    execution_time: float
    row_count: int
    columns: List[str]
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SQLiteHandler:
    """
    SQLite database handler for connection management and query execution
    """
    
    def __init__(self):
        self.connection = None
        self.db_path = None
        self.is_connected = False
        self.query_history = []  # Track executed queries
    
    def validate_database_file(self, db_path: str) -> Tuple[bool, str]:
        """
        Validate SQLite database file format and accessibility with security checks
        
        Args:
            db_path: Path to the SQLite database file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Enhanced security validation
            file_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
            is_secure, security_msg = security_validator.validate_database_file(db_path)
            
            if not is_secure:
                return False, security_msg
            
            return True, "Database file is valid and secure"
            
        except Exception as e:
            return False, f"Database validation failed: {str(e)}"
    
    def connect(self, db_path: str, password: str = None) -> bool:
        """
        Establish connection to SQLite database with connection pooling
        
        Args:
            db_path: Path to the SQLite database file
            password: Optional password for encrypted databases
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Validate database file first
            is_valid, error_msg = self.validate_database_file(db_path)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Close existing connection if any
            if self.connection:
                self.disconnect()
            
            # Use connection manager for optimized connections
            self.connection = connection_manager.get_connection(db_path)
            self.db_path = db_path
            self.is_connected = True
            
            # Test connection with a simple query
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            return True
            
        except Exception as e:
            self.is_connected = False
            self.connection = None
            self.db_path = None
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def disconnect(self):
        """Close database connection and cleanup resources"""
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None
        
        if self.db_path:
            # Close connection in connection manager
            try:
                connection_manager.close_connection(self.db_path)
            except:
                pass
            self.db_path = None
        
        self.is_connected = False
    
    @secure_query_execution
    def execute_query(self, query: str, params: tuple = None) -> QueryResult:
        """
        Execute SQL query with security validation, caching, and performance optimization
        
        Args:
            query: SQL query string
            params: Optional query parameters for parameterized queries
            
        Returns:
            QueryResult object containing DataFrame and metadata
        """
        if not self.is_connected:
            raise ConnectionError("No active database connection")
        
        import time
        start_time = time.time()
        
        try:
            # Additional validation (security validation is handled by decorator)
            is_valid, validation_msg = self.validate_query_syntax(query)
            if not is_valid:
                self.add_to_query_history(query, success=False, error_msg=validation_msg)
                raise ValueError(f"Query validation failed: {validation_msg}")
            
            # Check cache first (handled by decorator, but we need to handle params)
            cache_key = f"{query}_{str(params) if params else ''}"
            cached_result = performance_optimizer.get_cached_query_result(cache_key)
            
            if cached_result is not None:
                # Return cached result wrapped in QueryResult
                execution_time = 0.001  # Minimal time for cached result
                self.add_to_query_history(query, success=True)
                
                return QueryResult(
                    dataframe=cached_result,
                    query=query,
                    execution_time=execution_time,
                    row_count=len(cached_result),
                    columns=cached_result.columns.tolist(),
                    metadata={
                        'database_path': self.db_path,
                        'query_params': params,
                        'cached': True
                    }
                )
            
            # Execute query and fetch results
            if params:
                df = pd.read_sql_query(query, self.connection, params=params)
            else:
                df = pd.read_sql_query(query, self.connection)
            
            # Optimize DataFrame memory usage
            df = performance_optimizer.optimize_dataframe_memory(df)
            
            execution_time = time.time() - start_time
            
            # Cache the result
            performance_optimizer.cache_query_result(cache_key, df)
            
            # Add successful query to history
            self.add_to_query_history(query, success=True)
            
            # Create QueryResult object
            result = QueryResult(
                dataframe=df,
                query=query,
                execution_time=execution_time,
                row_count=len(df),
                columns=df.columns.tolist(),
                metadata={
                    'database_path': self.db_path,
                    'query_params': params,
                    'cached': False
                }
            )
            
            return result
            
        except Exception as e:
            # Add failed query to history
            error_msg = str(e)
            self.add_to_query_history(query, success=False, error_msg=error_msg)
            raise RuntimeError(f"Query execution failed: {error_msg}")
    
    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the database
        
        Returns:
            List of table names
        """
        if not self.is_connected:
            raise ConnectionError("No active database connection")
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve table names: {str(e)}")
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get row count for a specific table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows in the table
        """
        if not self.is_connected:
            raise ConnectionError("No active database connection")
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            count = cursor.fetchone()[0]
            return count
            
        except Exception as e:
            raise RuntimeError(f"Failed to get row count for table '{table_name}': {str(e)}")
    
    def get_table_schema(self, table_name: str) -> List[ColumnInfo]:
        """
        Get schema information for a specific table
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of ColumnInfo objects describing the table schema
        """
        if not self.is_connected:
            raise ConnectionError("No active database connection")
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            
            columns = []
            for row in cursor.fetchall():
                column_info = ColumnInfo(
                    name=row[1],
                    data_type=row[2],
                    nullable=not bool(row[3]),  # notnull flag is inverted
                    primary_key=bool(row[5]),
                    default_value=row[4]
                )
                columns.append(column_info)
            
            return columns
            
        except Exception as e:
            raise RuntimeError(f"Failed to get schema for table '{table_name}': {str(e)}")
    
    def get_table_info(self, table_name: str) -> TableInfo:
        """
        Get comprehensive information about a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            TableInfo object with complete table metadata
        """
        try:
            row_count = self.get_table_row_count(table_name)
            columns = self.get_table_schema(table_name)
            
            table_info = TableInfo(
                name=table_name,
                row_count=row_count,
                columns=columns
            )
            
            return table_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to get table info for '{table_name}': {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def get_foreign_keys(self, table_name: str) -> List[Dict]:
        """
        Get foreign key relationships for a specific table
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of dictionaries containing foreign key information
        """
        if not self.is_connected:
            raise ConnectionError("No active database connection")
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            
            foreign_keys = []
            for row in cursor.fetchall():
                fk_info = {
                    'id': row[0],
                    'seq': row[1],
                    'table': row[2],  # Referenced table
                    'from': row[3],   # Column in current table
                    'to': row[4],     # Column in referenced table
                    'on_update': row[5],
                    'on_delete': row[6],
                    'match': row[7]
                }
                foreign_keys.append(fk_info)
            
            return foreign_keys
            
        except Exception as e:
            raise RuntimeError(f"Failed to get foreign keys for table '{table_name}': {str(e)}")
    
    def get_table_preview(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Get preview of table data (first N rows)
        
        Args:
            table_name: Name of the table
            limit: Number of rows to return (default: 5)
            
        Returns:
            DataFrame containing the first N rows of the table
        """
        if not self.is_connected:
            raise ConnectionError("No active database connection")
        
        try:
            query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
            df = pd.read_sql_query(query, self.connection)
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to get preview for table '{table_name}': {str(e)}")
    
    def validate_query_syntax(self, query: str) -> Tuple[bool, str]:
        """
        Validate SQL query syntax with enhanced security checks
        
        Args:
            query: SQL query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.is_connected:
            return False, "No active database connection"
        
        try:
            # Use enhanced security validation
            is_secure, security_msg = security_validator.validate_sql_query(query)
            if not is_secure:
                return False, security_msg
            
            # Try to prepare the statement to check syntax
            cursor = self.connection.cursor()
            try:
                # Use EXPLAIN to validate syntax without executing
                cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                return True, "Query syntax is valid and secure"
            except sqlite3.Error as e:
                return False, f"SQL syntax error: {str(e)}"
            
        except Exception as e:
            return False, f"Error validating query: {str(e)}"
    
    def add_to_query_history(self, query: str, success: bool = True, error_msg: str = None):
        """
        Add query to history tracking
        
        Args:
            query: SQL query string
            success: Whether the query executed successfully
            error_msg: Error message if query failed
        """
        import datetime
        
        history_entry = {
            'timestamp': datetime.datetime.now(),
            'query': query,
            'success': success,
            'error_msg': error_msg
        }
        
        self.query_history.append(history_entry)
        
        # Keep only last 100 queries to prevent memory issues
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
    
    def get_query_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent query history
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of query history entries
        """
        return self.query_history[-limit:] if self.query_history else []
    
    def get_tables(self) -> List[TableInfo]:
        """
        Get comprehensive information about all tables in the database
        
        Returns:
            List of TableInfo objects for all tables
        """
        if not self.is_connected:
            raise ConnectionError("No active database connection")
        
        try:
            table_names = self.get_table_names()
            tables_info = []
            
            for table_name in table_names:
                try:
                    # Get basic table info
                    row_count = self.get_table_row_count(table_name)
                    columns = self.get_table_schema(table_name)
                    foreign_keys = self.get_foreign_keys(table_name)
                    
                    # Get indexes for the table
                    cursor = self.connection.cursor()
                    cursor.execute(f"PRAGMA index_list(`{table_name}`)")
                    indexes = [row[1] for row in cursor.fetchall()]
                    
                    table_info = TableInfo(
                        name=table_name,
                        row_count=row_count,
                        columns=columns,
                        foreign_keys=foreign_keys,
                        indexes=indexes
                    )
                    
                    tables_info.append(table_info)
                    
                except Exception as e:
                    # Log error but continue with other tables
                    print(f"Warning: Failed to get info for table '{table_name}': {str(e)}")
                    continue
            
            return tables_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to get tables information: {str(e)}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connection is closed"""
        self.disconnect()