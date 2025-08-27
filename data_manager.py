"""
Data Source Management Layer for DataSuperAgent

This module provides a unified interface for managing both file-based and database
data sources, enabling seamless switching between different data types.
"""

import pandas as pd
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid

from database_handler import SQLiteHandler, TableInfo


@dataclass
class DataSource:
    """Base data model for all data sources"""
    id: str
    name: str
    source_type: str  # 'file' or 'database'
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_accessed: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def update_access_time(self):
        """Update the last accessed timestamp"""
        self.last_accessed = datetime.datetime.now()


class DataSourceHandler(ABC):
    """Abstract base class for data source handlers"""
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from the source and return as DataFrame"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source"""
        pass
    
    @abstractmethod
    def validate_source(self) -> tuple[bool, str]:
        """Validate the data source and return (is_valid, error_message)"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources associated with the data source"""
        pass


class FileDataSource(DataSourceHandler):
    """Handler for CSV and Excel file data sources"""
    
    def __init__(self, file_path: str, file_type: str = None):
        self.file_path = Path(file_path)
        self.file_type = file_type or self._detect_file_type()
        self._dataframe = None
        
    def _detect_file_type(self) -> str:
        """Detect file type from extension"""
        extension = self.file_path.suffix.lower()
        if extension == '.csv':
            return 'csv'
        elif extension in ['.xlsx', '.xls']:
            return 'excel'
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file"""
        if self._dataframe is not None:
            return self._dataframe
            
        try:
            if self.file_type == 'csv':
                self._dataframe = pd.read_csv(self.file_path)
            elif self.file_type == 'excel':
                self._dataframe = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")
                
            return self._dataframe
            
        except Exception as e:
            raise RuntimeError(f"Failed to load file {self.file_path}: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get file metadata"""
        try:
            df = self.load_data()
            return {
                'file_path': str(self.file_path),
                'file_type': self.file_type,
                'file_size': self.file_path.stat().st_size,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        except Exception as e:
            return {'error': f"Failed to get metadata: {str(e)}"}
    
    def validate_source(self) -> tuple[bool, str]:
        """Validate file exists and is readable"""
        if not self.file_path.exists():
            return False, f"File does not exist: {self.file_path}"
        
        if not self.file_path.is_file():
            return False, f"Path is not a file: {self.file_path}"
        
        try:
            # Try to load a small sample to validate format
            if self.file_type == 'csv':
                pd.read_csv(self.file_path, nrows=1)
            elif self.file_type == 'excel':
                pd.read_excel(self.file_path, nrows=1)
            return True, "File is valid"
        except Exception as e:
            return False, f"File validation failed: {str(e)}"
    
    def cleanup(self):
        """Clean up cached data"""
        self._dataframe = None


class DatabaseDataSource(DataSourceHandler):
    """Handler for SQLite database data sources"""
    
    def __init__(self, db_path: str, table_name: str = None, query: str = None):
        self.db_path = db_path
        self.table_name = table_name
        self.query = query
        self.handler = SQLiteHandler()
        self._dataframe = None
        self._is_connected = False
        
        if not table_name and not query:
            raise ValueError("Either table_name or query must be provided")
    
    def connect(self) -> bool:
        """Connect to the database"""
        try:
            success = self.handler.connect(self.db_path)
            self._is_connected = success
            return success
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def load_data(self) -> pd.DataFrame:
        """Load data from database table or query"""
        if self._dataframe is not None:
            return self._dataframe
            
        if not self._is_connected:
            self.connect()
        
        try:
            if self.query:
                # Execute custom query
                result = self.handler.execute_query(self.query)
                self._dataframe = result.dataframe
            elif self.table_name:
                # Load entire table
                query = f"SELECT * FROM `{self.table_name}`"
                result = self.handler.execute_query(query)
                self._dataframe = result.dataframe
            else:
                raise ValueError("No table name or query specified")
                
            return self._dataframe
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from database: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get database metadata"""
        try:
            if not self._is_connected:
                self.connect()
            
            metadata = {
                'db_path': self.db_path,
                'source_type': 'database'
            }
            
            if self.table_name:
                table_info = self.handler.get_table_info(self.table_name)
                metadata.update({
                    'table_name': self.table_name,
                    'row_count': table_info.row_count,
                    'column_count': len(table_info.columns),
                    'columns': [col.name for col in table_info.columns],
                    'column_types': {col.name: col.data_type for col in table_info.columns},
                    'foreign_keys': table_info.foreign_keys,
                    'indexes': table_info.indexes
                })
            
            if self.query:
                metadata['custom_query'] = self.query
                # Get column info from loaded data
                df = self.load_data()
                metadata.update({
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict()
                })
            
            return metadata
            
        except Exception as e:
            return {'error': f"Failed to get metadata: {str(e)}"}
    
    def validate_source(self) -> tuple[bool, str]:
        """Validate database connection and table/query"""
        try:
            # Validate database file
            is_valid, error_msg = self.handler.validate_database_file(self.db_path)
            if not is_valid:
                return False, error_msg
            
            # Try to connect
            if not self._is_connected:
                self.connect()
            
            # Validate table exists or query is valid
            if self.table_name:
                tables = self.handler.get_table_names()
                if self.table_name not in tables:
                    return False, f"Table '{self.table_name}' does not exist"
            
            if self.query:
                is_valid, error_msg = self.handler.validate_query_syntax(self.query)
                if not is_valid:
                    return False, f"Invalid query: {error_msg}"
            
            return True, "Database source is valid"
            
        except Exception as e:
            return False, f"Database validation failed: {str(e)}"
    
    def cleanup(self):
        """Clean up database connection and cached data"""
        if self.handler:
            self.handler.disconnect()
        self._dataframe = None
        self._is_connected = False


class DataManager:
    """Unified manager for multiple data sources"""
    
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        self.source_handlers: Dict[str, DataSourceHandler] = {}
        self.active_source_id: Optional[str] = None
        
    def add_file_source(self, file_path: str, name: str = None) -> str:
        """Add a file-based data source"""
        source_id = str(uuid.uuid4())
        
        if name is None:
            name = Path(file_path).stem
        
        # Create file handler
        handler = FileDataSource(file_path)
        
        # Validate source
        is_valid, error_msg = handler.validate_source()
        if not is_valid:
            raise ValueError(f"Invalid file source: {error_msg}")
        
        # Get metadata
        metadata = handler.get_metadata()
        
        # Create data source
        data_source = DataSource(
            id=source_id,
            name=name,
            source_type='file',
            metadata=metadata
        )
        
        self.data_sources[source_id] = data_source
        self.source_handlers[source_id] = handler
        
        # Set as active if it's the first source
        if self.active_source_id is None:
            self.active_source_id = source_id
        
        return source_id
    
    def add_database_source(self, db_path: str, table_name: str = None, 
                          query: str = None, name: str = None) -> str:
        """Add a database-based data source"""
        source_id = str(uuid.uuid4())
        
        if name is None:
            if table_name:
                name = f"{Path(db_path).stem}.{table_name}"
            else:
                name = f"{Path(db_path).stem}.query"
        
        # Create database handler
        handler = DatabaseDataSource(db_path, table_name, query)
        
        # Validate source
        is_valid, error_msg = handler.validate_source()
        if not is_valid:
            raise ValueError(f"Invalid database source: {error_msg}")
        
        # Get metadata
        metadata = handler.get_metadata()
        
        # Create data source
        data_source = DataSource(
            id=source_id,
            name=name,
            source_type='database',
            metadata=metadata
        )
        
        self.data_sources[source_id] = data_source
        self.source_handlers[source_id] = handler
        
        # Set as active if it's the first source
        if self.active_source_id is None:
            self.active_source_id = source_id
        
        return source_id
    
    def get_active_dataset(self) -> Optional[pd.DataFrame]:
        """Get the currently active dataset"""
        if not self.active_source_id:
            return None
        
        try:
            handler = self.source_handlers[self.active_source_id]
            data_source = self.data_sources[self.active_source_id]
            data_source.update_access_time()
            return handler.load_data()
        except Exception as e:
            raise RuntimeError(f"Failed to load active dataset: {str(e)}")
    
    def switch_dataset(self, source_id: str) -> bool:
        """Switch to a different dataset"""
        if source_id not in self.data_sources:
            return False
        
        self.active_source_id = source_id
        self.data_sources[source_id].update_access_time()
        return True
    
    def get_dataset_metadata(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific dataset"""
        if source_id not in self.data_sources:
            return None
        
        data_source = self.data_sources[source_id]
        handler = self.source_handlers[source_id]
        
        # Combine data source info with handler metadata
        metadata = {
            'id': data_source.id,
            'name': data_source.name,
            'source_type': data_source.source_type,
            'created_at': data_source.created_at,
            'last_accessed': data_source.last_accessed,
            **data_source.metadata
        }
        
        return metadata
    
    def list_data_sources(self) -> List[Dict[str, Any]]:
        """Get list of all data sources with basic info"""
        sources = []
        for source_id, data_source in self.data_sources.items():
            source_info = {
                'id': source_id,
                'name': data_source.name,
                'source_type': data_source.source_type,
                'created_at': data_source.created_at,
                'last_accessed': data_source.last_accessed,
                'is_active': source_id == self.active_source_id
            }
            sources.append(source_info)
        
        return sorted(sources, key=lambda x: x['last_accessed'], reverse=True)
    
    def remove_data_source(self, source_id: str) -> bool:
        """Remove a data source"""
        if source_id not in self.data_sources:
            return False
        
        # Clean up handler resources
        handler = self.source_handlers[source_id]
        handler.cleanup()
        
        # Remove from collections
        del self.data_sources[source_id]
        del self.source_handlers[source_id]
        
        # Update active source if needed
        if self.active_source_id == source_id:
            if self.data_sources:
                # Set the most recently accessed source as active
                sources = self.list_data_sources()
                self.active_source_id = sources[0]['id'] if sources else None
            else:
                self.active_source_id = None
        
        return True
    
    def combine_datasets(self, source_ids: List[str], how: str = 'concat') -> pd.DataFrame:
        """Combine multiple datasets"""
        if not source_ids:
            raise ValueError("No source IDs provided")
        
        dataframes = []
        for source_id in source_ids:
            if source_id not in self.data_sources:
                raise ValueError(f"Source ID not found: {source_id}")
            
            handler = self.source_handlers[source_id]
            df = handler.load_data()
            
            # Add source identifier column
            df = df.copy()
            df['_source'] = self.data_sources[source_id].name
            dataframes.append(df)
        
        if how == 'concat':
            # Simple concatenation
            return pd.concat(dataframes, ignore_index=True)
        else:
            raise ValueError(f"Unsupported combination method: {how}")
    
    def check_schema_compatibility(self, source_ids: List[str]) -> Dict[str, Any]:
        """Check schema compatibility between datasets"""
        if len(source_ids) < 2:
            return {"compatible": True, "issues": [], "suggestions": []}
        
        compatibility_report = {
            "compatible": True,
            "issues": [],
            "suggestions": [],
            "common_columns": [],
            "column_conflicts": [],
            "merge_recommendations": []
        }
        
        try:
            # Get metadata for all sources
            source_metadata = {}
            for source_id in source_ids:
                if source_id in self.data_sources:
                    metadata = self.get_dataset_metadata(source_id)
                    source_metadata[source_id] = metadata
            
            # Find common columns
            all_columns = {}
            for source_id, metadata in source_metadata.items():
                if 'columns' in metadata:
                    source_name = metadata.get('name', source_id)
                    for col in metadata['columns']:
                        if col not in all_columns:
                            all_columns[col] = []
                        all_columns[col].append(source_name)
            
            # Identify common columns
            common_cols = {col: sources for col, sources in all_columns.items() 
                          if len(sources) >= 2}
            compatibility_report["common_columns"] = list(common_cols.keys())
            
            # Check for potential merge keys
            potential_keys = []
            for col in common_cols.keys():
                if any(keyword in col.lower() for keyword in ['id', 'key', 'code', 'number']):
                    potential_keys.append(col)
            
            if potential_keys:
                compatibility_report["merge_recommendations"] = [
                    f"Consider using '{key}' as a merge key" for key in potential_keys
                ]
            
            # Check data types compatibility (for database sources)
            type_conflicts = []
            for source_id, metadata in source_metadata.items():
                if 'column_types' in metadata:
                    for col, dtype in metadata['column_types'].items():
                        # Check if same column has different types in other sources
                        for other_id, other_metadata in source_metadata.items():
                            if (other_id != source_id and 
                                'column_types' in other_metadata and 
                                col in other_metadata['column_types']):
                                other_dtype = other_metadata['column_types'][col]
                                if dtype != other_dtype:
                                    type_conflicts.append({
                                        'column': col,
                                        'source1': metadata.get('name', source_id),
                                        'type1': dtype,
                                        'source2': other_metadata.get('name', other_id),
                                        'type2': other_dtype
                                    })
            
            if type_conflicts:
                compatibility_report["column_conflicts"] = type_conflicts
                compatibility_report["issues"].append("Data type conflicts found between sources")
                compatibility_report["compatible"] = False
            
            # Generate suggestions
            if len(compatibility_report["common_columns"]) == 0:
                compatibility_report["suggestions"].append(
                    "No common columns found. Consider adding identifier columns for joining."
                )
                compatibility_report["compatible"] = False
            elif len(compatibility_report["common_columns"]) < 3:
                compatibility_report["suggestions"].append(
                    "Limited common columns. Verify data alignment before combining."
                )
            
            if not potential_keys:
                compatibility_report["suggestions"].append(
                    "No obvious merge keys found. Manual column mapping may be required."
                )
            
        except Exception as e:
            compatibility_report["compatible"] = False
            compatibility_report["issues"].append(f"Error checking compatibility: {str(e)}")
        
        return compatibility_report
    
    def suggest_data_combinations(self) -> List[Dict[str, Any]]:
        """Suggest potential data combinations based on schema analysis"""
        suggestions = []
        
        if len(self.data_sources) < 2:
            return suggestions
        
        source_ids = list(self.data_sources.keys())
        
        # Check all pairs of sources
        for i in range(len(source_ids)):
            for j in range(i + 1, len(source_ids)):
                source1_id = source_ids[i]
                source2_id = source_ids[j]
                
                compatibility = self.check_schema_compatibility([source1_id, source2_id])
                
                if compatibility["compatible"] or compatibility["common_columns"]:
                    source1_name = self.data_sources[source1_id].name
                    source2_name = self.data_sources[source2_id].name
                    
                    suggestion = {
                        "source1_id": source1_id,
                        "source2_id": source2_id,
                        "source1_name": source1_name,
                        "source2_name": source2_name,
                        "compatibility_score": len(compatibility["common_columns"]),
                        "common_columns": compatibility["common_columns"],
                        "merge_recommendations": compatibility["merge_recommendations"],
                        "combination_type": "join" if compatibility["merge_recommendations"] else "concat"
                    }
                    suggestions.append(suggestion)
        
        # Sort by compatibility score
        suggestions.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def create_cross_dataset_query(self, source_ids: List[str], join_column: str = None) -> str:
        """Generate SQL-like query for cross-dataset analysis"""
        if len(source_ids) < 2:
            return ""
        
        query_parts = []
        
        # Get source information
        sources_info = []
        for source_id in source_ids:
            if source_id in self.data_sources:
                metadata = self.get_dataset_metadata(source_id)
                sources_info.append({
                    'id': source_id,
                    'name': metadata.get('name', source_id),
                    'type': metadata.get('source_type', 'unknown'),
                    'columns': metadata.get('columns', [])
                })
        
        if join_column:
            # Generate JOIN query
            base_source = sources_info[0]
            query_parts.append(f"-- Cross-dataset JOIN analysis")
            query_parts.append(f"SELECT *")
            query_parts.append(f"FROM {base_source['name']} t1")
            
            for i, source in enumerate(sources_info[1:], 2):
                query_parts.append(f"JOIN {source['name']} t{i} ON t1.{join_column} = t{i}.{join_column}")
        else:
            # Generate UNION query
            query_parts.append(f"-- Cross-dataset UNION analysis")
            for i, source in enumerate(sources_info):
                if i > 0:
                    query_parts.append("UNION ALL")
                query_parts.append(f"SELECT *, '{source['name']}' as source_dataset FROM {source['name']}")
        
        return "\n".join(query_parts)
    
    def cleanup_all(self):
        """Clean up all data sources"""
        for handler in self.source_handlers.values():
            handler.cleanup()
        
        self.data_sources.clear()
        self.source_handlers.clear()
        self.active_source_id = None


def generate_profile_report(df: pd.DataFrame, title: str = "Data Profile Report"):
    """
    Generates a data profile report using ydata-profiling.

    Args:
        df: The DataFrame to profile.
        title: The title for the profile report.

    Returns:
        A ydata_profiling.ProfileReport object.
    """
    from ydata_profiling import ProfileReport

    if df is None or df.empty:
        return None

    try:
        profile = ProfileReport(
            df,
            title=title,
            minimal=True,  # Use minimal mode for performance in a web app
            explorative=True,
            dark_mode=True,
            orange_mode=True,
        )
        return profile
    except Exception as e:
        print(f"Error generating profile report: {e}")
        return None