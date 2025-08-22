"""
Comprehensive Error Handling and User Feedback System for DataSuperAgent

This module provides centralized error handling, user-friendly error messages,
progress indicators, and help system for the application.
"""

import streamlit as st
import traceback
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
import time
from contextlib import contextmanager


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.error_log = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataSuperAgent')
    
    def log_error(self, error: Exception, context: str = "", user_action: str = ""):
        """Log error with context information"""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'user_action': user_action,
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_info)
        self.logger.error(f"{context}: {error_info['error_message']}")
        
        # Keep only last 50 errors to prevent memory issues
        if len(self.error_log) > 50:
            self.error_log = self.error_log[-50:]
    
    def get_user_friendly_message(self, error: Exception, context: str = "") -> Dict[str, str]:
        """Convert technical errors to user-friendly messages"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Database-related errors
        if 'sqlite' in error_msg or 'database' in error_msg:
            if 'no such table' in error_msg:
                return {
                    'title': '🗄️ Table Not Found',
                    'message': 'The specified table does not exist in the database.',
                    'suggestion': 'Please check the table name and try again. You can view available tables in the database schema section.',
                    'action': 'Check available tables'
                }
            elif 'syntax error' in error_msg:
                return {
                    'title': '📝 SQL Syntax Error',
                    'message': 'There is a syntax error in your SQL query.',
                    'suggestion': 'Please check your SQL syntax. Use the query validation feature before executing.',
                    'action': 'Validate query syntax'
                }
            elif 'locked' in error_msg:
                return {
                    'title': '🔒 Database Locked',
                    'message': 'The database file is currently locked by another process.',
                    'suggestion': 'Please close any other applications using this database and try again.',
                    'action': 'Retry connection'
                }
            elif 'permission' in error_msg or 'access' in error_msg:
                return {
                    'title': '🚫 Access Denied',
                    'message': 'Permission denied when accessing the database file.',
                    'suggestion': 'Please check file permissions and ensure you have read access to the database.',
                    'action': 'Check file permissions'
                }
        
        # File-related errors
        elif 'file' in error_msg or error_type in ['FileNotFoundError', 'PermissionError']:
            if 'not found' in error_msg or error_type == 'FileNotFoundError':
                return {
                    'title': '📁 File Not Found',
                    'message': 'The specified file could not be found.',
                    'suggestion': 'Please check the file path and ensure the file exists.',
                    'action': 'Select a different file'
                }
            elif 'permission' in error_msg or error_type == 'PermissionError':
                return {
                    'title': '🔐 Permission Error',
                    'message': 'Permission denied when accessing the file.',
                    'suggestion': 'Please check file permissions and ensure you have read access.',
                    'action': 'Check file permissions'
                }
        
        # Memory-related errors
        elif 'memory' in error_msg or error_type == 'MemoryError':
            return {
                'title': '💾 Memory Error',
                'message': 'The operation requires more memory than available.',
                'suggestion': 'Try working with a smaller dataset or use data sampling.',
                'action': 'Reduce data size'
            }
        
        # Network/API errors
        elif 'api' in error_msg or 'network' in error_msg or 'connection' in error_msg:
            return {
                'title': '🌐 Connection Error',
                'message': 'Failed to connect to the AI service.',
                'suggestion': 'Please check your internet connection and API key configuration.',
                'action': 'Check connection'
            }
        
        # Data format errors
        elif 'pandas' in error_msg or 'dataframe' in error_msg:
            return {
                'title': '📊 Data Format Error',
                'message': 'There is an issue with the data format or structure.',
                'suggestion': 'Please check your data for missing values, incorrect types, or formatting issues.',
                'action': 'Review data format'
            }
        
        # Generic error
        return {
            'title': '⚠️ Unexpected Error',
            'message': f'An unexpected error occurred: {str(error)}',
            'suggestion': 'Please try again or contact support if the problem persists.',
            'action': 'Retry operation'
        }
    
    def display_error(self, error: Exception, context: str = "", user_action: str = ""):
        """Display user-friendly error message in Streamlit"""
        self.log_error(error, context, user_action)
        
        error_info = self.get_user_friendly_message(error, context)
        
        st.error(f"**{error_info['title']}**")
        st.write(error_info['message'])
        
        if error_info['suggestion']:
            st.info(f"💡 **Suggestion:** {error_info['suggestion']}")
        
        # Show detailed error in expander for debugging
        with st.expander("🔧 Technical Details (for debugging)", expanded=False):
            st.write(f"**Error Type:** {type(error).__name__}")
            st.write(f"**Context:** {context}")
            st.write(f"**User Action:** {user_action}")
            st.code(str(error))
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        if not self.error_log:
            return {'total_errors': 0, 'error_types': {}, 'recent_errors': []}
        
        error_types = {}
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recent_errors': self.error_log[-5:]  # Last 5 errors
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(context: str = "", user_action: str = ""):
    """Decorator for automatic error handling"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.display_error(e, context, user_action)
                return None
        return wrapper
    return decorator


@contextmanager
def progress_indicator(message: str, success_message: str = ""):
    """Context manager for showing progress indicators"""
    progress_placeholder = st.empty()
    
    with progress_placeholder:
        with st.spinner(message):
            try:
                yield
                if success_message:
                    st.success(success_message)
            except Exception as e:
                error_handler.display_error(e, "Progress operation", message)
                raise
            finally:
                # Clear the progress indicator after a short delay
                time.sleep(0.5)
                progress_placeholder.empty()


class HelpSystem:
    """Comprehensive help and documentation system"""
    
    @staticmethod
    def show_database_help():
        """Show database-specific help"""
        st.markdown("""
        ### 🗄️ Database Features Help
        
        **Supported Database Types:**
        - SQLite (.db, .sqlite, .sqlite3)
        
        **Getting Started:**
        1. **Upload Database:** Use the "Database (SQLite)" tab in Step 1
        2. **Select Tables:** Choose which tables to analyze
        3. **Preview Data:** Review table structure and sample data
        4. **Load Data:** Load selected tables or execute custom queries
        
        **SQL Query Interface:**
        - Only SELECT queries are allowed for security
        - Use query validation before execution
        - Save frequently used queries for reuse
        - View query history and results
        
        **Table Relationships:**
        - Foreign keys are automatically detected
        - Relationship information is shown in table previews
        - AI agent is aware of table relationships
        
        **Common Issues:**
        - **Database locked:** Close other applications using the database
        - **Table not found:** Check table names in the schema view
        - **Permission denied:** Ensure file read permissions
        """)
    
    @staticmethod
    def show_dataset_management_help():
        """Show dataset management help"""
        st.markdown("""
        ### 📊 Dataset Management Help
        
        **Multiple Data Sources:**
        - Load files (CSV/Excel) and databases simultaneously
        - Switch between different datasets easily
        - Each dataset maintains its own chat history
        
        **Dataset Switching:**
        - Use the sidebar dropdown to switch active datasets
        - Chat history is preserved per dataset
        - AI agent context updates automatically
        
        **Cross-Dataset Analysis:**
        - Compare schemas between datasets
        - Combine compatible datasets
        - Generate cross-dataset queries
        - View compatibility reports
        
        **Data Combination:**
        - **Concatenate:** Stack rows from multiple datasets
        - **Schema Comparison:** Compare column structures
        - **Join Analysis:** Find common columns for joining
        """)
    
    @staticmethod
    def show_export_help():
        """Show export functionality help"""
        st.markdown("""
        ### 📤 Export Features Help
        
        **Export Formats:**
        - **CSV:** Data with optional metadata comments
        - **Excel:** Multi-sheet export with metadata and analysis history
        - **JSON:** Structured data with full metadata
        - **SQL:** Query scripts for database sources
        
        **Export Options:**
        - **Include Metadata:** Source information and schema details
        - **Include Analysis History:** Chat history and insights
        - **Anonymize Data:** Replace sensitive information
        
        **Database-Specific Exports:**
        - Original table information
        - Foreign key relationships
        - Custom query used to generate data
        - SQL scripts for recreating queries
        """)
    
    @staticmethod
    def show_troubleshooting():
        """Show troubleshooting guide"""
        st.markdown("""
        ### 🔧 Troubleshooting Guide
        
        **Common Issues and Solutions:**
        
        **Database Problems:**
        - **"Database is locked"** → Close other database applications
        - **"Table not found"** → Check table names in schema view
        - **"SQL syntax error"** → Use query validation feature
        - **"Permission denied"** → Check file permissions
        
        **File Upload Issues:**
        - **"File format not supported"** → Use CSV, Excel, or SQLite files
        - **"File too large"** → Try smaller files or data sampling
        - **"Encoding error"** → Ensure UTF-8 encoding
        
        **AI Agent Issues:**
        - **"Agent not ready"** → Complete Step 3 to prepare agent
        - **"No response"** → Check API key configuration
        - **"Memory error"** → Use smaller datasets
        
        **Performance Issues:**
        - **Slow queries** → Add LIMIT clauses to large queries
        - **High memory usage** → Work with data samples
        - **Slow UI** → Refresh the page or restart application
        
        **Getting Help:**
        - Check technical details in error messages
        - Review the error log in the sidebar
        - Ensure all requirements are installed
        """)


def show_help_sidebar():
    """Show help options in sidebar"""
    with st.sidebar.expander("❓ Help & Support", expanded=False):
        help_topic = st.selectbox(
            "Select help topic:",
            [
                "Database Features",
                "Dataset Management", 
                "Export Features",
                "Troubleshooting",
                "Error Statistics"
            ],
            key="help_topic_selector"
        )
        
        if help_topic == "Database Features":
            HelpSystem.show_database_help()
        elif help_topic == "Dataset Management":
            HelpSystem.show_dataset_management_help()
        elif help_topic == "Export Features":
            HelpSystem.show_export_help()
        elif help_topic == "Troubleshooting":
            HelpSystem.show_troubleshooting()
        elif help_topic == "Error Statistics":
            stats = error_handler.get_error_statistics()
            st.write(f"**Total Errors:** {stats['total_errors']}")
            if stats['error_types']:
                st.write("**Error Types:**")
                for error_type, count in stats['error_types'].items():
                    st.write(f"• {error_type}: {count}")


def validate_operation(operation_name: str, validation_func: Callable, *args, **kwargs):
    """Validate operation before execution with user feedback"""
    try:
        with st.spinner(f"Validating {operation_name}..."):
            result = validation_func(*args, **kwargs)
            if result:
                st.success(f"✅ {operation_name} validation passed")
                return True
            else:
                st.warning(f"⚠️ {operation_name} validation failed")
                return False
    except Exception as e:
        error_handler.display_error(e, f"{operation_name} validation", "Validating operation")
        return False


def safe_execute(func: Callable, context: str = "", success_message: str = "", *args, **kwargs):
    """Safely execute function with comprehensive error handling"""
    try:
        with progress_indicator(f"Executing {context}...", success_message):
            result = func(*args, **kwargs)
            return result
    except Exception as e:
        error_handler.display_error(e, context, "Executing operation")
        return None