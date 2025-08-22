import streamlit as st
import os
import sys
import atexit
from dotenv import load_dotenv
from ui import main_ui, sidebar_ui
from security_performance import cleanup_resources

# Configure Streamlit page
st.set_page_config(
    page_title="DataSuperAgent - Smart Data Analysis", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
TEMP_PLOT_DIR = "temp_plots"
if not os.path.exists(TEMP_PLOT_DIR):
    os.makedirs(TEMP_PLOT_DIR)

# Load environment variables
load_dotenv()

def initialize_application():
    """Initialize application components and check dependencies"""
    
    # Check for required API key
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("🚨 GOOGLE_API_KEY environment variable not set!")
        st.info("Please set the GOOGLE_API_KEY environment variable before running.")
        with st.expander("How to set up API Key", expanded=True):
            st.markdown("""
            1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Create a `.env` file in your project directory
            3. Add the line: `GOOGLE_API_KEY=your_api_key_here`
            4. Restart the application
            """)
        st.stop()
    
    # Initialize session state for database components
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = True
        
        # Initialize data manager
        try:
            from data_manager import DataManager
            if "data_manager" not in st.session_state:
                st.session_state.data_manager = DataManager()
        except ImportError as e:
            st.error(f"Failed to import data manager: {e}")
            st.info("Some database features may not be available.")
        
        # Initialize database handler check
        try:
            from database_handler import SQLiteHandler
            # Test database handler functionality
            test_handler = SQLiteHandler()
        except ImportError as e:
            st.error(f"Failed to import database handler: {e}")
            st.info("Database features will not be available.")
        except Exception as e:
            st.warning(f"Database handler initialization warning: {e}")
        
        # Initialize enhanced agent handler
        try:
            from agent_handler import create_agent
        except ImportError as e:
            st.error(f"Failed to import agent handler: {e}")
            st.stop()
        
        # Backward compatibility check
        try:
            from data_handler import load_and_combine_files
            from utils import get_llm
        except ImportError as e:
            st.error(f"Failed to import required components: {e}")
            st.info("Please ensure all required files are present.")
            st.stop()

def check_system_requirements():
    """Check system requirements and display warnings if needed"""
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 7):
        st.warning(f"⚠️ Python {python_version.major}.{python_version.minor} detected. Python 3.7+ recommended.")
    
    # Check for required packages
    required_packages = [
        'streamlit', 'pandas', 'langchain', 'langchain_experimental', 
        'langchain_google_genai', 'sqlite3', 'openpyxl'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
        st.info("Please install missing packages using: `pip install -r requirements.txt`")
        st.stop()

def display_welcome_message():
    """Display welcome message and feature overview"""
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = True
        
        with st.container():
            st.markdown("""
            ### 🎉 Welcome to DataSuperAgent Enhanced!
            
            **New Features:**
            - 🗄️ **SQLite Database Support** - Upload and analyze database files
            - 🔍 **SQL Query Interface** - Execute custom queries with validation
            - 📊 **Multi-Dataset Management** - Switch between different data sources
            - 🔗 **Cross-Dataset Analysis** - Combine and compare multiple datasets
            - 🤖 **Enhanced AI Context** - Database-aware AI responses
            - 📤 **Advanced Export** - Export with metadata and analysis history
            
            **Getting Started:**
            1. Upload CSV/Excel files OR SQLite databases in Step 1
            2. Load and explore your data in Step 2
            3. Prepare the AI agent in Step 3
            4. Start chatting with your data in Step 4
            """)
            
            if st.button("Got it! Let's start analyzing 🚀"):
                st.session_state.welcome_shown = True
                st.rerun()

def main():
    """Main application entry point"""
    
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    # Initialize application
    initialize_application()
    
    # Check system requirements
    check_system_requirements()
    
    # Display welcome message for new users
    if not st.session_state.get("welcome_shown", False):
        display_welcome_message()
        return
    
    # Run main UI components
    try:
        main_ui()
        sidebar_ui()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or check the console for more details.")
        
        # Show error details in expander for debugging
        with st.expander("Error Details (for debugging)", expanded=False):
            import traceback
            st.code(traceback.format_exc())
    
    # Cleanup on session end
    if st.session_state.get("cleanup_on_exit", False):
        cleanup_resources()

if __name__ == "__main__":
    main()
