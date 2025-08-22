import streamlit as st
import os
import re
import time
import traceback
import pandas as pd
import io
import tempfile
from agent_handler import create_agent
from data_handler import load_and_combine_files
from utils import get_llm, find_potential_common_columns
from error_handler import error_handler, handle_errors, progress_indicator, safe_execute, show_help_sidebar
from security_performance import monitor_performance, security_validator

TEMP_PLOT_DIR = "temp_plots"
TEMP_PLOT_FILE = os.path.abspath(os.path.join(TEMP_PLOT_DIR, "temp_plot.png"))

def get_followup_suggestions(prompt, response, df_columns, _llm):
    """Generates follow-up question suggestions using the suggestion LLM."""
    if not _llm: return []
    try:
        cleaned_response = response[:500]
        columns_str = ", ".join(df_columns)
        suggestion_prompt = f"""
        Given the user's question: "{prompt}"
        And the chatbot's answer: "{cleaned_response}"
        The available data columns are: [{columns_str}]

        Suggest 3 concise and relevant follow-up questions the user might ask next. Format them clearly:
        1. Suggestion 1?
        2. Suggestion 2?
        3. Suggestion 3?
        """
        suggestion_response = _llm.invoke(suggestion_prompt)
        suggestions_text = suggestion_response.content

        if 'session_llm_calls' not in st.session_state:
            st.session_state.session_llm_calls = 0
        if 'session_estimated_tokens' not in st.session_state:
            st.session_state.session_estimated_tokens = 0.0

        st.session_state.session_llm_calls += 1
        prompt_len = len(str(suggestion_prompt if suggestion_prompt is not None else ""))
        response_len = len(str(suggestions_text if suggestions_text is not None else ""))
        st.session_state.session_estimated_tokens += (prompt_len + response_len) / 4.0

        suggestions = []
        potential_suggestions = re.findall(r"^\s*\d+\.\s+(.*)", suggestions_text, re.MULTILINE)
        suggestions = [s.strip("? ").strip() + "?" for s in potential_suggestions[:3]]

        if not suggestions and len(suggestions_text.splitlines()) > 1:
             lines = suggestions_text.split('\n')
             suggestions = [line.strip("? ").strip() + "?" for line in lines if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '*')][:3]

        return suggestions
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return []

def main_ui():
    st.title("🧠 Smart Data Analysis Agent ✨")
    st.caption(f"Upload CSV/Excel files or SQLite databases, and chat with an AI agent about the data.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "combined_df" not in st.session_state:
        st.session_state.combined_df = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = False
    if "current_prompt" not in st.session_state:
         st.session_state.current_prompt = ""
    if "uploaded_files_processed" not in st.session_state:
        st.session_state.uploaded_files_processed = False
    if 'session_llm_calls' not in st.session_state:
        st.session_state.session_llm_calls = 0
    if 'session_estimated_tokens' not in st.session_state:
        st.session_state.session_estimated_tokens = 0.0
    if "dataset_chat_histories" not in st.session_state:
        st.session_state.dataset_chat_histories = {}
    
    # Handle chat history switching when dataset changes
    if "data_manager" in st.session_state and st.session_state.data_manager.active_source_id:
        active_source_id = st.session_state.data_manager.active_source_id
        
        # Save current chat history to previous dataset
        if "previous_active_source" in st.session_state and st.session_state.previous_active_source != active_source_id:
            if st.session_state.messages and st.session_state.previous_active_source:
                st.session_state.dataset_chat_histories[st.session_state.previous_active_source] = st.session_state.messages.copy()
        
        # Load chat history for current dataset
        if active_source_id in st.session_state.dataset_chat_histories:
            st.session_state.messages = st.session_state.dataset_chat_histories[active_source_id].copy()
        else:
            # New dataset, start fresh
            if "previous_active_source" in st.session_state and st.session_state.previous_active_source != active_source_id:
                st.session_state.messages = []
        
        st.session_state.previous_active_source = active_source_id

    # Initialize data manager if not exists
    if "data_manager" not in st.session_state:
        from data_manager import DataManager
        st.session_state.data_manager = DataManager()
    
    with st.expander("Step 1: Upload Data Files", expanded=True):
        tab1, tab2 = st.tabs(["📄 Files (CSV/Excel)", "🗄️ Database (SQLite)"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Choose CSV or Excel files",
                accept_multiple_files=True,
                type=['csv', 'xlsx', 'xls'],
                key="file_uploader"
            )
            if uploaded_files:
                st.session_state.uploaded_files_processed = False
        
        with tab2:
            uploaded_db = st.file_uploader(
                "Choose SQLite database file",
                accept_multiple_files=False,
                type=['db', 'sqlite', 'sqlite3'],
                key="db_uploader"
            )
            
            if uploaded_db:
                # Save uploaded database to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                    tmp_file.write(uploaded_db.read())
                    temp_db_path = tmp_file.name
                
                try:
                    # Validate and show database info with progress indicator
                    with progress_indicator("Validating database file...", "Database validated successfully"):
                        from database_handler import SQLiteHandler
                        db_handler = SQLiteHandler()
                        is_valid, error_msg = db_handler.validate_database_file(temp_db_path)
                        
                        if is_valid:
                            st.success(f"✅ Valid SQLite database: {uploaded_db.name}")
                            
                            # Connect and show tables with error handling
                            with progress_indicator("Loading database schema...", "Schema loaded successfully"):
                                db_handler.connect(temp_db_path)
                                tables = db_handler.get_tables()
                                
                                if tables:
                                    st.write(f"**Found {len(tables)} tables:**")
                                    
                                    # Enhanced table selection interface
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write("**Table Selection:**")
                                        
                                        # Multi-select for tables
                                        table_options = {}
                                        for table in tables:
                                            display_name = f"{table.name} ({table.row_count} rows)"
                                            table_options[display_name] = table.name
                                        
                                        selected_tables_display = st.multiselect(
                                            "Select tables to explore:",
                                            options=list(table_options.keys()),
                                            key="table_multiselect"
                                        )
                                        
                                        if selected_tables_display:
                                            selected_tables = [table_options[display] for display in selected_tables_display]
                                            
                                            # Single table selection for detailed view
                                            if len(selected_tables) > 1:
                                                primary_table = st.selectbox(
                                                    "Primary table for preview:",
                                                    options=selected_tables,
                                                    key="primary_table_selector"
                                                )
                                            else:
                                                primary_table = selected_tables[0]
                                    
                                    with col2:
                                        if selected_tables_display:
                                            st.write("**Table Information:**")
                                            
                                            # Show table stats
                                            for table_name in selected_tables:
                                                table_info = next(t for t in tables if t.name == table_name)
                                                with st.expander(f"📊 {table_name}", expanded=table_name==primary_table):
                                                    
                                                    # Basic stats
                                                    st.metric("Rows", table_info.row_count)
                                                    st.metric("Columns", len(table_info.columns))
                                                    
                                                    # Foreign key relationships
                                                    if table_info.foreign_keys:
                                                        st.write("**🔗 Foreign Keys:**")
                                                        for fk in table_info.foreign_keys:
                                                            st.write(f"• {fk['from']} → {fk['table']}.{fk['to']}")
                                                    
                                                    # Indexes
                                                    if table_info.indexes:
                                                        st.write("**📇 Indexes:**")
                                                        for idx in table_info.indexes:
                                                            st.write(f"• {idx}")
                                    
                                    # Detailed preview for primary table
                                    if selected_tables_display and primary_table:
                                        st.write("---")
                                        st.write(f"**📋 Preview: {primary_table}**")
                                        
                                        # Table preview with more options
                                        preview_rows = st.slider("Preview rows:", 1, 20, 5, key="preview_slider")
                                        preview_df = db_handler.get_table_preview(primary_table, limit=preview_rows)
                                        st.dataframe(preview_df, use_container_width=True)
                                        
                                        # Schema information
                                        with st.expander("🏗️ Schema Details", expanded=False):
                                            table_info = next(t for t in tables if t.name == primary_table)
                                            schema_data = []
                                            for col in table_info.columns:
                                                schema_data.append({
                                                    'Column': col.name,
                                                    'Type': col.data_type,
                                                    'Nullable': '✅' if col.nullable else '❌',
                                                    'Primary Key': '🔑' if col.primary_key else '',
                                                    'Default': col.default_value or ''
                                                })
                                            st.dataframe(pd.DataFrame(schema_data), use_container_width=True)
                                        
                                        # Load options
                                        st.write("**Load Options:**")
                                        load_option = st.radio(
                                            "How would you like to load the data?",
                                            ["Single Table", "Multiple Tables (Combined)", "Custom Query"],
                                            key="load_option_radio"
                                        )
                                        
                                        if load_option == "Single Table":
                                            if st.button("Load Selected Table", key="load_single_table_btn"):
                                                try:
                                                    source_id = st.session_state.data_manager.add_database_source(
                                                        temp_db_path, 
                                                        table_name=primary_table,
                                                        name=f"{uploaded_db.name}.{primary_table}"
                                                    )
                                                    st.success(f"✅ Table '{primary_table}' loaded successfully!")
                                                    st.session_state.uploaded_files_processed = False
                                                    st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                                                    
                                                except Exception as e:
                                                    st.error(f"Failed to load table: {str(e)}")
                                        
                                                        elif load_option == "Multiple Tables (Combined)" and len(selected_tables) > 1:
                                            st.info("Multiple tables will be loaded as separate data sources that can be combined later.")
                                            if st.button("Load All Selected Tables", key="load_multiple_tables_btn"):
                                                try:
                                                    loaded_sources = []
                                                    for table_name in selected_tables:
                                                        source_id = st.session_state.data_manager.add_database_source(
                                                            temp_db_path, 
                                                            table_name=table_name,
                                                            name=f"{uploaded_db.name}.{table_name}"
                                                        )
                                                        loaded_sources.append(source_id)
                                                    
                                                    st.success(f"✅ Loaded {len(loaded_sources)} tables successfully!")
                                                    st.session_state.uploaded_files_processed = False
                                                    st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                                                    
                                                except Exception as e:
                                                    st.error(f"Failed to load tables: {str(e)}")
                                        
                                        elif load_option == "Custom Query":
                                    st.write("**SQL Query Editor:**")
                                    
                                    # Sample queries
                                    sample_queries = {
                                        "Select All": f"SELECT * FROM {primary_table}",
                                        "Count Rows": f"SELECT COUNT(*) as total_rows FROM {primary_table}",
                                        "First 100 Rows": f"SELECT * FROM {primary_table} LIMIT 100"
                                    }
                                    
                                    if len(selected_tables) > 1:
                                        # Add join example if there are foreign keys
                                        for table_name in selected_tables:
                                            table_info = next(t for t in tables if t.name == table_name)
                                            for fk in table_info.foreign_keys:
                                                if fk['table'] in selected_tables:
                                                    sample_queries["Join Example"] = f"""SELECT * FROM {table_name} t1 
JOIN {fk['table']} t2 ON t1.{fk['from']} = t2.{fk['to']} 
LIMIT 100"""
                                                    break
                                    
                                    query_template = st.selectbox(
                                        "Query templates:",
                                        options=["Custom"] + list(sample_queries.keys()),
                                        key="query_template_selector"
                                    )
                                    
                                    if query_template != "Custom":
                                        default_query = sample_queries[query_template]
                                    else:
                                        default_query = f"SELECT * FROM {primary_table} LIMIT 100"
                                    
                                    custom_query = st.text_area(
                                        "SQL Query:",
                                        value=default_query,
                                        height=100,
                                        key="custom_query_input"
                                    )
                                    
                                    col_validate, col_load = st.columns([1, 1])
                                    
                                    with col_validate:
                                        if st.button("Validate Query", key="validate_query_btn"):
                                            is_valid, msg = db_handler.validate_query_syntax(custom_query)
                                            if is_valid:
                                                st.success("✅ Query is valid!")
                                            else:
                                                st.error(f"❌ {msg}")
                                    
                                    with col_load:
                                        if st.button("Load Query Results", key="load_query_btn"):
                                            try:
                                                # Validate first
                                                is_valid, msg = db_handler.validate_query_syntax(custom_query)
                                                if not is_valid:
                                                    st.error(f"Invalid query: {msg}")
                                                else:
                                                    source_id = st.session_state.data_manager.add_database_source(
                                                        temp_db_path, 
                                                        query=custom_query,
                                                        name=f"{uploaded_db.name}.custom_query"
                                                    )
                                                    st.success("✅ Query results loaded successfully!")
                                                    st.session_state.uploaded_files_processed = False
                                                    st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                                                    
                                            except Exception as e:
                                                st.error(f"Failed to load query results: {str(e)}")
                                else:
                                    st.warning("No tables found in the database.")
                                
                                db_handler.disconnect()
                        else:
                            st.error(f"❌ Invalid database file: {error_msg}")
                
                except Exception as e:
                    st.error(f"Error processing database: {str(e)}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_db_path)
                    except:
                        pass

    # SQL Query Interface for loaded databases
    if "data_manager" in st.session_state:
        db_sources = [s for s in st.session_state.data_manager.list_data_sources() if s['source_type'] == 'database']
        if db_sources:
            with st.expander("🔍 SQL Query Interface", expanded=False):
                st.write("**Execute custom SQL queries on your loaded databases**")
                
                # Select database source
                db_source_options = {f"{s['name']} (ID: {s['id'][:8]}...)": s['id'] for s in db_sources}
                selected_db_display = st.selectbox(
                    "Select database source:",
                    options=list(db_source_options.keys()),
                    key="sql_db_selector"
                )
                
                if selected_db_display:
                    selected_db_id = db_source_options[selected_db_display]
                    db_metadata = st.session_state.data_manager.get_dataset_metadata(selected_db_id)
                    
                    # Show available tables for reference
                    if 'db_path' in db_metadata:
                        try:
                            from database_handler import SQLiteHandler
                            temp_handler = SQLiteHandler()
                            temp_handler.connect(db_metadata['db_path'])
                            available_tables = temp_handler.get_table_names()
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**SQL Query Editor:**")
                                
                                # Query history
                                query_history = temp_handler.get_query_history()
                                if query_history:
                                    with st.expander("📜 Query History", expanded=False):
                                        for i, entry in enumerate(reversed(query_history[-10:])):  # Show last 10
                                            status_icon = "✅" if entry['success'] else "❌"
                                            timestamp = entry['timestamp'].strftime("%H:%M:%S")
                                            st.write(f"{status_icon} **{timestamp}** - {entry['query'][:50]}...")
                                            if not entry['success'] and entry.get('error_msg'):
                                                st.caption(f"Error: {entry['error_msg']}")
                                
                                # Query templates
                                query_templates = {
                                    "Custom Query": "",
                                    "Show All Tables": "SELECT name FROM sqlite_master WHERE type='table'",
                                    "Table Info": f"SELECT * FROM {available_tables[0] if available_tables else 'table_name'} LIMIT 10",
                                    "Count Records": f"SELECT COUNT(*) as total FROM {available_tables[0] if available_tables else 'table_name'}",
                                }
                                
                                # Add join templates if multiple tables
                                if len(available_tables) > 1:
                                    query_templates["Join Tables"] = f"""SELECT * FROM {available_tables[0]} t1 
JOIN {available_tables[1]} t2 ON t1.id = t2.id 
LIMIT 10"""
                                
                                template_choice = st.selectbox(
                                    "Query Templates:",
                                    options=list(query_templates.keys()),
                                    key="sql_template_selector"
                                )
                                
                                # SQL input area
                                sql_query = st.text_area(
                                    "Enter your SQL query:",
                                    value=query_templates.get(template_choice, ""),
                                    height=150,
                                    key="sql_query_input",
                                    help="Only SELECT queries are allowed for security reasons"
                                )
                                
                                # Query controls
                                col_validate, col_execute, col_save = st.columns([1, 1, 1])
                                
                                with col_validate:
                                    if st.button("🔍 Validate", key="sql_validate_btn"):
                                        if sql_query.strip():
                                            is_valid, msg = temp_handler.validate_query_syntax(sql_query)
                                            if is_valid:
                                                st.success("✅ Valid query!")
                                            else:
                                                st.error(f"❌ {msg}")
                                        else:
                                            st.warning("Please enter a query")
                                
                                with col_execute:
                                    if st.button("▶️ Execute", key="sql_execute_btn"):
                                        if sql_query.strip():
                                            def execute_sql_query():
                                                result = temp_handler.execute_query(sql_query)
                                                return result
                                            
                                            result = safe_execute(
                                                execute_sql_query,
                                                context="SQL query execution",
                                                success_message=f"✅ Query executed successfully!"
                                            )
                                            
                                            if result:
                                                st.write(f"**Results: {result.row_count} rows** (Execution time: {result.execution_time:.3f}s)")
                                                    
                                                    # Show results
                                                    if not result.dataframe.empty:
                                                        # Limit display for large results
                                                        display_limit = 1000
                                                        if len(result.dataframe) > display_limit:
                                                            st.warning(f"Showing first {display_limit} rows of {len(result.dataframe)} total rows")
                                                            st.dataframe(result.dataframe.head(display_limit), use_container_width=True)
                                                        else:
                                                            st.dataframe(result.dataframe, use_container_width=True)
                                                        
                                                        # Store result in session for potential use
                                                        st.session_state.last_query_result = result.dataframe
                                                        
                                                        # Option to load as new data source
                                                        if st.button("💾 Load Results as New Dataset", key="load_query_results_btn"):
                                                            try:
                                                                source_id = st.session_state.data_manager.add_database_source(
                                                                    db_metadata['db_path'],
                                                                    query=sql_query,
                                                                    name=f"Query_{len(st.session_state.data_manager.data_sources)+1}"
                                                                )
                                                                st.success("✅ Query results loaded as new dataset!")
                                                                st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                                                            except Exception as e:
                                                                st.error(f"Failed to load results: {str(e)}")
                                                    else:
                                                        st.info("Query returned no results")
                                                        

                                        else:
                                            st.warning("Please enter a query")
                                
                                with col_save:
                                    if st.button("💾 Save Query", key="sql_save_btn"):
                                        if sql_query.strip():
                                            # Save to session state for reuse
                                            if "saved_queries" not in st.session_state:
                                                st.session_state.saved_queries = []
                                            
                                            query_name = f"Query_{len(st.session_state.saved_queries)+1}"
                                            st.session_state.saved_queries.append({
                                                'name': query_name,
                                                'query': sql_query,
                                                'timestamp': time.time()
                                            })
                                            st.success(f"✅ Query saved as '{query_name}'")
                                        else:
                                            st.warning("Please enter a query to save")
                                
                                # Saved queries
                                if "saved_queries" in st.session_state and st.session_state.saved_queries:
                                    with st.expander("💾 Saved Queries", expanded=False):
                                        for saved_query in st.session_state.saved_queries:
                                            col_name, col_load = st.columns([3, 1])
                                            with col_name:
                                                st.write(f"**{saved_query['name']}**")
                                                st.code(saved_query['query'][:100] + "..." if len(saved_query['query']) > 100 else saved_query['query'])
                                            with col_load:
                                                if st.button("Load", key=f"load_saved_{saved_query['name']}"):
                                                    st.session_state.sql_query_input = saved_query['query']
                                                    st.rerun()
                            
                            with col2:
                                st.write("**Database Schema:**")
                                
                                # Show tables and their schemas
                                for table_name in available_tables:
                                    with st.expander(f"📋 {table_name}", expanded=False):
                                        try:
                                            table_info = temp_handler.get_table_info(table_name)
                                            st.write(f"**Rows:** {table_info.row_count}")
                                            
                                            # Column info
                                            st.write("**Columns:**")
                                            for col in table_info.columns:
                                                pk_indicator = " 🔑" if col.primary_key else ""
                                                st.write(f"• `{col.name}` ({col.data_type}){pk_indicator}")
                                            
                                            # Foreign keys
                                            if table_info.foreign_keys:
                                                st.write("**Foreign Keys:**")
                                                for fk in table_info.foreign_keys:
                                                    st.write(f"• `{fk['from']}` → `{fk['table']}.{fk['to']}`")
                                        
                                        except Exception as e:
                                            st.error(f"Error loading table info: {str(e)}")
                            
                            temp_handler.disconnect()
                            
                        except Exception as e:
                            st.error(f"Error connecting to database: {str(e)}")

    with st.expander("Step 2: Load and Combine Data"):
        # Handle traditional file uploads (backward compatibility)
        if uploaded_files and not st.session_state.uploaded_files_processed:
            if st.button("Load and Combine Uploaded Files"):
                try:
                    # Use new data manager if available, fallback to old method
                    if "data_manager" in st.session_state:
                        # New method: Add files to data manager with progress tracking
                        loaded_sources = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            progress = (i + 1) / len(uploaded_files)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                            
                            # Validate file before processing
                            file_size = len(uploaded_file.read())
                            uploaded_file.seek(0)  # Reset file pointer
                            
                            # Create secure temporary file
                            temp_file_path = security_validator.create_secure_temp_file(
                                suffix=f".{uploaded_file.name.split('.')[-1]}"
                            )
                            
                            # Validate file security
                            is_safe, safety_msg = security_validator.validate_file_upload(
                                uploaded_file.name, file_size
                            )
                            
                            if not is_safe:
                                st.error(f"File validation failed for {uploaded_file.name}: {safety_msg}")
                                continue
                            
                            # Save uploaded file to secure temp location
                            with open(temp_file_path, 'wb') as temp_file:
                                temp_file.write(uploaded_file.read())
                            
                            def load_file():
                                return st.session_state.data_manager.add_file_source(
                                    temp_file_path, 
                                    name=uploaded_file.name
                                )
                            
                            source_id = safe_execute(
                                load_file,
                                context=f"Loading file {uploaded_file.name}",
                                success_message=""
                            )
                            
                            if source_id:
                                loaded_sources.append(source_id)
                            
                            # Clean up temp file
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if loaded_sources:
                            if len(loaded_sources) > 1:
                                # Combine multiple files
                                combined_df = st.session_state.data_manager.combine_datasets(loaded_sources)
                                
                                # Create combined source
                                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                                combined_df.to_csv(temp_file.name, index=False)
                                temp_file.close()
                                
                                combined_source_id = st.session_state.data_manager.add_file_source(
                                    temp_file.name,
                                    name=f"Combined_{len(uploaded_files)}_files"
                                )
                                
                                st.info(f"Multiple files combined into single dataset. Total rows: {len(combined_df)}")
                            
                            # Set combined_df for backward compatibility
                            st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                            st.success(f"✅ Loaded {len(uploaded_files)} file(s) successfully!")
                        else:
                            st.error("Failed to load any files.")
                            st.session_state.combined_df = None
                    
                    else:
                        # Fallback to old method
                        combined_df_result = load_and_combine_files(uploaded_files)
                        if combined_df_result is not None:
                            if isinstance(combined_df_result, list):
                                if not combined_df_result:
                                    st.warning("No data was loaded from the files.")
                                    st.session_state.combined_df = None
                                elif len(combined_df_result) == 1 and isinstance(combined_df_result[0], dict) and 'df' in combined_df_result[0] and isinstance(combined_df_result[0]['df'], pd.DataFrame):
                                    st.session_state.combined_df = combined_df_result[0]['df']
                                elif all(isinstance(item, dict) and 'df' in item and isinstance(item['df'], pd.DataFrame) for item in combined_df_result):
                                    try:
                                        dfs_to_concat = [item['df'] for item in combined_df_result if item.get('df') is not None and not item['df'].empty]
                                        if dfs_to_concat:
                                            st.session_state.combined_df = pd.concat(dfs_to_concat, ignore_index=True)
                                            st.info(f"Multiple files were loaded and have been concatenated into a single DataFrame for analysis. Total rows: {len(st.session_state.combined_df)}")
                                        else:
                                            st.warning("No valid DataFrames found to concatenate from the loaded files.")
                                            st.session_state.combined_df = None
                                    except Exception as e:
                                        st.error(f"Error concatenating multiple DataFrames: {str(e)}")
                                        st.session_state.combined_df = None
                                else:
                                    st.error("Loaded data is in an unexpected list format. Cannot prepare for agent analysis.")
                                    st.session_state.combined_df = None
                            elif isinstance(combined_df_result, pd.DataFrame):
                                st.session_state.combined_df = combined_df_result
                            else:
                                st.error("Loaded data is not in a recognizable DataFrame or list format.")
                                st.session_state.combined_df = None
                        else:
                            st.error("Failed to load or combine data.")
                            st.session_state.combined_df = None
                    
                    # Reset agent and mark as processed
                    st.session_state.agent = None
                    st.session_state.agent_ready = False
                    st.session_state.messages = []
                    st.session_state.uploaded_files_processed = True
                    
                    # Show preview
                    if st.session_state.combined_df is not None:
                        st.write("Preview of Combined Data (first 5 rows):")
                        st.dataframe(st.session_state.combined_df.head())
                    
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")
                    st.session_state.combined_df = None
                    st.session_state.uploaded_files_processed = False

        elif st.session_state.combined_df is not None:
             st.success("Data is loaded and ready for analysis.")
             
             # Show current dataset info
             if "data_manager" in st.session_state and st.session_state.data_manager.active_source_id:
                 metadata = st.session_state.data_manager.get_dataset_metadata(
                     st.session_state.data_manager.active_source_id
                 )
                 if metadata:
                     col1, col2, col3 = st.columns(3)
                     with col1:
                         st.metric("Active Dataset", metadata.get('name', 'Unknown'))
                     with col2:
                         st.metric("Rows", f"{metadata.get('row_count', 0):,}")
                     with col3:
                         st.metric("Columns", metadata.get('column_count', 0))
             
             st.write("Preview of Active Data (first 5 rows):")
             if isinstance(st.session_state.combined_df, pd.DataFrame):
                 st.dataframe(st.session_state.combined_df.head())
             else:
                 st.warning("Data is not in the expected DataFrame format.")
        else:
             st.info("Upload files or databases in Step 1 and load them here.")

    with st.expander("Step 3: Prepare AI Agent for Analysis"):
        if st.session_state.combined_df is not None and not st.session_state.agent_ready:
            if st.button("Prepare Agent"):
                @handle_errors(context="Agent preparation", user_action="Preparing AI agent")
                def prepare_agent():
                    with progress_indicator("Initializing AI agent...", "AI Agent prepared successfully!"):
                        llm_agent_model = get_llm(model_name="gemini-2.0-flash", temperature=0)
                        
                        # Get metadata for current active dataset
                        metadata = None
                        if "data_manager" in st.session_state and st.session_state.data_manager.active_source_id:
                            metadata = st.session_state.data_manager.get_dataset_metadata(
                                st.session_state.data_manager.active_source_id
                            )
                        
                        agent = create_agent(st.session_state.combined_df, llm_agent_model, metadata)
                        if agent is None:
                            raise RuntimeError("Failed to create AI agent")
                        
                        st.session_state.agent = agent
                        st.session_state.agent_ready = True
                        
                        # Show context-aware success message
                        if metadata and metadata.get('source_type') == 'database':
                            if 'table_name' in metadata:
                                st.success(f"✅ AI Agent is ready with database context for table '{metadata['table_name']}'!")
                            else:
                                st.success("✅ AI Agent is ready with database context!")
                            
                            # Show database-specific suggestions
                            from agent_handler import get_database_suggestions
                            suggestions = get_database_suggestions(metadata)
                            if suggestions:
                                st.info("💡 **Suggested analyses for this database table:**")
                                for i, suggestion in enumerate(suggestions[:3], 1):
                                    st.write(f"{i}. {suggestion}")
                        else:
                            st.success("✅ AI Agent is ready! Proceed to Step 4 to chat.")
                        
                        return True
                
                prepare_agent()

        elif st.session_state.agent_ready:
             st.success("✅ AI Agent is ready! Proceed to Step 4 to chat.")
        else:
             st.info("Load and combine data in Step 2 first.")

    st.divider()
    st.subheader("Step 4: Chat with your Data Agent")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("plot_path") and os.path.exists(message["plot_path"]):
                 st.image(message["plot_path"])
            elif message.get("plot_path"):
                 st.caption("[Plot image not found - may have been cleared]")

    if st.session_state.current_prompt:
        prompt = st.session_state.current_prompt
        st.session_state.current_prompt = ""
    else:
        prompt = st.chat_input("Ask the agent about the loaded data...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "plot_path": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.agent is None or not st.session_state.agent_ready:
            st.warning("⚠️ Agent is not ready. Please complete Step 3 first.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Agent is not ready. Please prepare the agent in Step 3.",
                "plot_path": None
            })
        else:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking and executing..."):
                    message_placeholder = st.empty()
                    plot_path_this_turn = None
                    if os.path.exists(TEMP_PLOT_FILE):
                        try:
                            os.remove(TEMP_PLOT_FILE)
                            print(f"Cleared previous plot file: {TEMP_PLOT_FILE}")
                        except Exception as e_rem:
                            print(f"Warning: Could not remove previous plot file: {e_rem}")

                    try:
                        print(f"\n>>> Running Agent with Input:\n{prompt}\n")
                        
                        # Execute agent with timeout and error handling
                        with st.spinner("🤖 AI is analyzing your data..."):
                            response = st.session_state.agent.invoke({"input": f"{prompt}"})
                            final_answer = response.get("output") if isinstance(response, dict) else response
                        
                        print(f"\n<<< Agent Raw Response:\n{final_answer}\n")

                        st.session_state.session_llm_calls += 1
                        prompt_len = len(str(prompt if prompt is not None else ""))
                        response_len = len(str(final_answer if final_answer is not None else ""))
                        st.session_state.session_estimated_tokens += (prompt_len + response_len) / 4.0

                        time.sleep(0.5)
                        if os.path.exists(TEMP_PLOT_FILE):
                            print(f"Confirmed plot file exists after agent run: {TEMP_PLOT_FILE}")
                            plot_path_this_turn = TEMP_PLOT_FILE
                            message_placeholder.markdown(final_answer)
                            st.image(plot_path_this_turn)
                        else:
                            print(f"Plot file NOT found after agent run at: {TEMP_PLOT_FILE}")
                            message_placeholder.markdown(final_answer)

                        suggestions = []
                        if "error executing code" not in final_answer.lower() and st.session_state.combined_df is not None:
                            llm_suggestion_model = get_llm(model_name="gemini-2.0-flash", temperature=0.7)
                            
                            # Get metadata for context-aware suggestions
                            metadata = None
                            if "data_manager" in st.session_state and st.session_state.data_manager.active_source_id:
                                metadata = st.session_state.data_manager.get_dataset_metadata(
                                    st.session_state.data_manager.active_source_id
                                )
                            
                            # Use enhanced follow-up suggestions if we have database context
                            if metadata and metadata.get('source_type') == 'database':
                                from agent_handler import create_context_aware_followup_prompt
                                enhanced_prompt = create_context_aware_followup_prompt(prompt, final_answer, metadata)
                                
                                try:
                                    suggestion_response = llm_suggestion_model.invoke(enhanced_prompt)
                                    suggestions_text = suggestion_response.content
                                    
                                    # Parse suggestions
                                    import re
                                    potential_suggestions = re.findall(r"^\s*\d+\.\s+(.*)", suggestions_text, re.MULTILINE)
                                    suggestions = [s.strip("? ").strip() + "?" for s in potential_suggestions[:3]]
                                    
                                    if not suggestions and len(suggestions_text.splitlines()) > 1:
                                        lines = suggestions_text.split('\n')
                                        suggestions = [line.strip("? ").strip() + "?" for line in lines 
                                                     if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '*')][:3]
                                    
                                    # Update token count
                                    st.session_state.session_llm_calls += 1
                                    prompt_len = len(str(enhanced_prompt))
                                    response_len = len(str(suggestions_text))
                                    st.session_state.session_estimated_tokens += (prompt_len + response_len) / 4.0
                                    
                                except Exception as e:
                                    print(f"Error generating enhanced suggestions: {e}")
                                    # Fallback to regular suggestions
                                    suggestions = get_followup_suggestions(
                                        prompt, final_answer,
                                        st.session_state.combined_df.columns.tolist(),
                                        llm_suggestion_model
                                    )
                            else:
                                # Use regular suggestions for file-based data
                                suggestions = get_followup_suggestions(
                                    prompt, final_answer,
                                    st.session_state.combined_df.columns.tolist(),
                                    llm_suggestion_model
                                )

                        if suggestions:
                            st.markdown("---")
                            st.markdown("**Suggested follow-up questions:**")
                            cols = st.columns(len(suggestions))
                            for i, sugg in enumerate(suggestions):
                                button_key = f"suggestion_{len(st.session_state.messages)}_{i}"
                                if cols[i].button(sugg, key=button_key, use_container_width=True):
                                    st.session_state.current_prompt = sugg
                                    st.rerun()
                    except Exception as e:
                        # Enhanced error handling for agent execution
                        error_handler.display_error(e, "AI Agent execution", f"Processing query: {prompt[:50]}...")
                        
                        # Provide helpful suggestions based on error type
                        error_msg = str(e).lower()
                        if 'api' in error_msg or 'key' in error_msg:
                            st.info("💡 **Tip:** Check your Google API key configuration in the .env file")
                        elif 'memory' in error_msg:
                            st.info("💡 **Tip:** Try working with a smaller dataset or simpler query")
                        elif 'timeout' in error_msg:
                            st.info("💡 **Tip:** The query might be too complex. Try breaking it into smaller parts")
                        
                        final_answer = "I encountered an error while processing your request. Please check the error details above and try again."
                        suggestions = []
                        
                        # Clean up plot file if exists
                        if os.path.exists(TEMP_PLOT_FILE):
                             try: 
                                 os.remove(TEMP_PLOT_FILE)
                             except Exception as e_rem_err: 
                                 print(f"Error removing plot file after exception: {e_rem_err}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "plot_path": plot_path_this_turn
            })

def sidebar_ui():
    st.sidebar.header("Options")
    
    # Add help system
    show_help_sidebar()
    
    # Add performance monitoring
    monitor_performance()
    
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.current_prompt = ""
        st.session_state.combined_df = None
        st.session_state.agent = None
        st.session_state.agent_ready = False
        st.session_state.uploaded_files_processed = False

        # Clean up data manager
        if "data_manager" in st.session_state:
            st.session_state.data_manager.cleanup_all()

        st.session_state.session_llm_calls = 0
        st.session_state.session_estimated_tokens = 0.0

        if os.path.exists(TEMP_PLOT_FILE):
           try:
               os.remove(TEMP_PLOT_FILE)
               print(f"Cleared plot file on session reset: {TEMP_PLOT_FILE}")
           except Exception as e_rem:
               print(f"Warning: Could not remove plot file on session reset: {e_rem}")

        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("📊 Dataset Management")
    
    # Show active data sources with management options
    if "data_manager" in st.session_state:
        sources = st.session_state.data_manager.list_data_sources()
        if sources:
            st.sidebar.write("**Active Data Sources:**")
            
            # Dataset switcher
            current_active = next((s for s in sources if s['is_active']), None)
            if current_active:
                st.sidebar.success(f"🟢 Active: {current_active['name']}")
            
            # Dataset selection dropdown
            source_options = {f"{s['source_type'].title()}: {s['name']}": s['id'] for s in sources}
            
            if len(sources) > 1:
                selected_source_display = st.sidebar.selectbox(
                    "Switch to dataset:",
                    options=["Keep Current"] + list(source_options.keys()),
                    key="dataset_switcher"
                )
                
                if selected_source_display != "Keep Current":
                    selected_source_id = source_options[selected_source_display]
                    if st.sidebar.button("🔄 Switch Dataset", key="switch_dataset_btn"):
                        if st.session_state.data_manager.switch_dataset(selected_source_id):
                            st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                            st.session_state.agent = None  # Reset agent for new dataset
                            st.session_state.agent_ready = False
                            
                            # Load chat history for switched dataset
                            if selected_source_id in st.session_state.dataset_chat_histories:
                                st.session_state.messages = st.session_state.dataset_chat_histories[selected_source_id].copy()
                            else:
                                st.session_state.messages = []
                            
                            st.success(f"Switched to: {selected_source_display}")
                            st.rerun()
                        else:
                            st.error("Failed to switch dataset")
            
            # Dataset information cards
            with st.sidebar.expander("📋 Dataset Details", expanded=False):
                for source in sources:
                    status_icon = "🟢" if source['is_active'] else "⚪"
                    source_type_icon = "📄" if source['source_type'] == 'file' else "🗄️"
                    
                    st.write(f"{status_icon} {source_type_icon} **{source['name']}**")
                    
                    # Get detailed metadata
                    metadata = st.session_state.data_manager.get_dataset_metadata(source['id'])
                    if metadata:
                        st.caption(f"Type: {metadata.get('source_type', 'Unknown')}")
                        st.caption(f"Rows: {metadata.get('row_count', 'Unknown')}")
                        st.caption(f"Columns: {metadata.get('column_count', 'Unknown')}")
                        st.caption(f"Created: {source['created_at'].strftime('%H:%M:%S')}")
                        
                        # Show columns
                        if 'columns' in metadata and metadata['columns']:
                            with st.expander(f"Columns ({len(metadata['columns'])})", expanded=False):
                                for col in metadata['columns'][:10]:  # Show first 10 columns
                                    st.write(f"• {col}")
                                if len(metadata['columns']) > 10:
                                    st.caption(f"... and {len(metadata['columns']) - 10} more")
                        
                        # Database-specific info
                        if source['source_type'] == 'database':
                            if 'table_name' in metadata:
                                st.caption(f"Table: {metadata['table_name']}")
                            if 'foreign_keys' in metadata and metadata['foreign_keys']:
                                st.caption(f"Foreign Keys: {len(metadata['foreign_keys'])}")
                            if 'custom_query' in metadata:
                                st.caption("Source: Custom Query")
                    
                    # Remove dataset option
                    if st.button(f"🗑️ Remove", key=f"remove_{source['id'][:8]}"):
                        if st.session_state.data_manager.remove_data_source(source['id']):
                            # Update combined_df if this was the active source
                            st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                            if st.session_state.combined_df is None:
                                st.session_state.agent = None
                                st.session_state.agent_ready = False
                            st.rerun()
                    
                    st.write("---")
            
            # Cross-dataset analysis capabilities
            if len(sources) > 1:
                with st.sidebar.expander("🔗 Cross-Dataset Analysis", expanded=False):
                    st.write("**Analyze relationships between datasets:**")
                    
                    # Show compatibility suggestions
                    suggestions = st.session_state.data_manager.suggest_data_combinations()
                    if suggestions:
                        st.write("**🎯 Recommended Combinations:**")
                        for i, suggestion in enumerate(suggestions[:3]):
                            with st.expander(f"{suggestion['source1_name']} + {suggestion['source2_name']}", expanded=False):
                                st.write(f"**Compatibility Score:** {suggestion['compatibility_score']}")
                                if suggestion['common_columns']:
                                    st.write(f"**Common Columns:** {', '.join(suggestion['common_columns'][:5])}")
                                if suggestion['merge_recommendations']:
                                    st.write(f"**Merge Keys:** {', '.join(suggestion['merge_recommendations'])}")
                                
                                # Quick combine button
                                if st.button(f"Combine These", key=f"quick_combine_{i}"):
                                    try:
                                        selected_ids = [suggestion['source1_id'], suggestion['source2_id']]
                                        combined_df = st.session_state.data_manager.combine_datasets(selected_ids)
                                        
                                        # Create new temporary data source
                                        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                                        combined_df.to_csv(temp_file.name, index=False)
                                        temp_file.close()
                                        
                                        source_id = st.session_state.data_manager.add_file_source(
                                            temp_file.name,
                                            name=f"Combined_{suggestion['source1_name']}_{suggestion['source2_name']}"
                                        )
                                        
                                        st.success("✅ Datasets combined!")
                                        st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                                        st.session_state.agent = None
                                        st.session_state.agent_ready = False
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"Failed to combine: {str(e)}")
                    
                    st.write("---")
                    st.write("**🔧 Manual Combination:**")
                    
                    # Multi-select for datasets to analyze
                    combine_options = {f"{s['name']}": s['id'] for s in sources}
                    selected_for_analysis = st.multiselect(
                        "Select datasets to analyze:",
                        options=list(combine_options.keys()),
                        key="analysis_datasets_selector"
                    )
                    
                    if len(selected_for_analysis) >= 2:
                        selected_ids = [combine_options[name] for name in selected_for_analysis]
                        
                        # Check compatibility
                        compatibility = st.session_state.data_manager.check_schema_compatibility(selected_ids)
                        
                        # Show compatibility report
                        if compatibility["compatible"]:
                            st.success("✅ Datasets are compatible!")
                        else:
                            st.warning("⚠️ Compatibility issues found")
                        
                        if compatibility["common_columns"]:
                            st.write(f"**Common columns:** {', '.join(compatibility['common_columns'][:10])}")
                        
                        if compatibility["issues"]:
                            st.write("**Issues:**")
                            for issue in compatibility["issues"]:
                                st.write(f"• {issue}")
                        
                        if compatibility["column_conflicts"]:
                            st.write("**Type conflicts:**")
                            for conflict in compatibility["column_conflicts"][:3]:
                                st.write(f"• `{conflict['column']}`: {conflict['type1']} vs {conflict['type2']}")
                        
                        # Combination options
                        combine_method = st.radio(
                            "Analysis method:",
                            ["Concatenate (Stack rows)", "Schema Comparison", "Generate Cross-Query"],
                            key="analysis_method_radio"
                        )
                        
                        if combine_method == "Concatenate (Stack rows)":
                            if st.button("🔗 Combine Selected", key="combine_datasets_btn"):
                                try:
                                    combined_df = st.session_state.data_manager.combine_datasets(selected_ids)
                                    
                                    # Create new temporary data source
                                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                                    combined_df.to_csv(temp_file.name, index=False)
                                    temp_file.close()
                                    
                                    source_id = st.session_state.data_manager.add_file_source(
                                        temp_file.name,
                                        name=f"Combined_{len(selected_for_analysis)}_datasets"
                                    )
                                    
                                    st.success(f"✅ Combined {len(selected_for_analysis)} datasets!")
                                    st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                                    st.session_state.agent = None
                                    st.session_state.agent_ready = False
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Failed to combine datasets: {str(e)}")
                        
                        elif combine_method == "Schema Comparison":
                            if st.button("📊 Compare Schemas", key="compare_schemas_btn"):
                                st.write("**Schema Comparison Report:**")
                                
                                # Create comparison table
                                all_columns = set()
                                source_columns = {}
                                
                                for source_id in selected_ids:
                                    metadata = st.session_state.data_manager.get_dataset_metadata(source_id)
                                    source_name = metadata.get('name', source_id)
                                    columns = metadata.get('columns', [])
                                    column_types = metadata.get('column_types', {})
                                    
                                    source_columns[source_name] = {
                                        'columns': columns,
                                        'types': column_types
                                    }
                                    all_columns.update(columns)
                                
                                # Create comparison DataFrame
                                comparison_data = []
                                for col in sorted(all_columns):
                                    row = {'Column': col}
                                    for source_name, info in source_columns.items():
                                        if col in info['columns']:
                                            col_type = info['types'].get(col, 'Unknown')
                                            row[source_name] = f"✅ ({col_type})"
                                        else:
                                            row[source_name] = "❌"
                                    comparison_data.append(row)
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                        
                        elif combine_method == "Generate Cross-Query":
                            join_column = None
                            if compatibility["common_columns"]:
                                join_column = st.selectbox(
                                    "Select join column:",
                                    options=["None (Union)"] + compatibility["common_columns"],
                                    key="join_column_selector"
                                )
                                if join_column == "None (Union)":
                                    join_column = None
                            
                            if st.button("🔍 Generate Query", key="generate_query_btn"):
                                query = st.session_state.data_manager.create_cross_dataset_query(
                                    selected_ids, join_column
                                )
                                st.write("**Generated Cross-Dataset Query:**")
                                st.code(query, language="sql")
                                
                                # Option to save query
                                if "saved_cross_queries" not in st.session_state:
                                    st.session_state.saved_cross_queries = []
                                
                                if st.button("💾 Save Query", key="save_cross_query_btn"):
                                    st.session_state.saved_cross_queries.append({
                                        'query': query,
                                        'datasets': selected_for_analysis,
                                        'timestamp': time.time()
                                    })
                                    st.success("Query saved!")
            
            # Chat history management per dataset
            with st.sidebar.expander("💬 Chat History Management", expanded=False):
                st.write("**Chat history is automatically managed per dataset**")
                
                if "dataset_chat_histories" not in st.session_state:
                    st.session_state.dataset_chat_histories = {}
                
                active_source_id = st.session_state.data_manager.active_source_id
                if active_source_id:
                    # Save current chat to active dataset
                    if st.session_state.messages:
                        st.session_state.dataset_chat_histories[active_source_id] = st.session_state.messages.copy()
                    
                    # Show chat history stats
                    for source in sources:
                        source_id = source['id']
                        chat_count = len(st.session_state.dataset_chat_histories.get(source_id, []))
                        status = "🟢" if source['is_active'] else "⚪"
                        st.write(f"{status} {source['name']}: {chat_count} messages")
                
                if st.button("🗑️ Clear All Chat Histories", key="clear_all_chats_btn"):
                    st.session_state.dataset_chat_histories = {}
                    st.session_state.messages = []
                    st.success("All chat histories cleared!")
                    st.rerun()
        
        else:
            st.sidebar.write("No data sources loaded")
            st.sidebar.info("Upload files or databases in Step 1 to get started")
    
    st.sidebar.divider()
    st.sidebar.header("Status & Debug Info")
    st.sidebar.write(f"Agent Ready: {'✅ Yes' if st.session_state.agent_ready else '❌ No'}")

    if 'combined_df' in st.session_state and st.session_state.combined_df is not None:
        st.sidebar.write("Combined DataFrame Info:")

        actual_df_to_display = None
        data_source_note = ""

        if isinstance(st.session_state.combined_df, pd.DataFrame):
            actual_df_to_display = st.session_state.combined_df
        elif isinstance(st.session_state.combined_df, list):
            if st.session_state.combined_df and isinstance(st.session_state.combined_df[0], dict) and 'df' in st.session_state.combined_df[0] and isinstance(st.session_state.combined_df[0]['df'], pd.DataFrame):
                actual_df_to_display = st.session_state.combined_df[0]['df']
                data_source_note = "(Info for first DataFrame in list)"
            else:
                st.sidebar.warning("Data is an unexpected list format.")
        else:
            st.sidebar.warning("Data is not a recognized DataFrame.")

        if actual_df_to_display is not None:
            if data_source_note:
                st.sidebar.caption(data_source_note)
            st.sidebar.write(f"- Rows: {len(actual_df_to_display)}")
            st.sidebar.write(f"- Columns: {len(actual_df_to_display.columns)}")
            with st.sidebar.expander("Show Column Names"):
                st.code(actual_df_to_display.columns.tolist())
            with st.sidebar.expander("Show Head (First 5 Rows)"):
                st.dataframe(actual_df_to_display.head())
    else:
        st.sidebar.warning("No data loaded/combined yet.")

    st.sidebar.divider()
    st.sidebar.subheader("📤 Export Data & Analysis")

    if 'combined_df' in st.session_state and isinstance(st.session_state.combined_df, pd.DataFrame) and not st.session_state.combined_df.empty:
        
        # Get metadata for enhanced export
        metadata = None
        if "data_manager" in st.session_state and st.session_state.data_manager.active_source_id:
            metadata = st.session_state.data_manager.get_dataset_metadata(
                st.session_state.data_manager.active_source_id
            )
        
        # Export options
        export_options = st.sidebar.multiselect(
            "Export options:",
            ["Include Metadata", "Include Analysis History", "Anonymize Data"],
            default=["Include Metadata"],
            key="export_options"
        )
        
        # Prepare enhanced data for export
        export_df = st.session_state.combined_df.copy()
        
        # Anonymization option
        if "Anonymize Data" in export_options:
            # Simple anonymization - replace text columns with generic values
            for col in export_df.columns:
                if export_df[col].dtype == 'object':
                    if export_df[col].astype(str).str.contains('@').any():  # Email-like
                        export_df[col] = f"email_{export_df.index + 1}@example.com"
                    elif export_df[col].astype(str).str.len().mean() > 10:  # Long text
                        export_df[col] = f"text_data_{export_df.index + 1}"
        
        # Create metadata info for export
        export_metadata = {}
        if "Include Metadata" in export_options and metadata:
            export_metadata = {
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_type": metadata.get('source_type', 'unknown'),
                "data_source": metadata.get('name', 'unknown'),
                "original_row_count": metadata.get('row_count', len(export_df)),
                "exported_row_count": len(export_df),
                "columns": list(export_df.columns),
            }
            
            if metadata.get('source_type') == 'database':
                export_metadata.update({
                    "database_path": metadata.get('db_path', ''),
                    "table_name": metadata.get('table_name', ''),
                    "custom_query": metadata.get('custom_query', ''),
                    "foreign_keys": metadata.get('foreign_keys', []),
                    "column_types": metadata.get('column_types', {})
                })
            elif metadata.get('source_type') == 'file':
                export_metadata.update({
                    "file_path": metadata.get('file_path', ''),
                    "file_type": metadata.get('file_type', '')
                })
        
        # Analysis history
        analysis_history = []
        if "Include Analysis History" in export_options and st.session_state.messages:
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    analysis_history.append({
                        "type": "question",
                        "content": msg['content'],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                elif msg['role'] == 'assistant':
                    analysis_history.append({
                        "type": "analysis",
                        "content": msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'],
                        "had_visualization": msg.get('plot_path') is not None
                    })
        
        # CSV Export with metadata
        try:
            csv_buffer = io.StringIO()
            
            # Write metadata as comments if included
            if export_metadata:
                csv_buffer.write("# DataSuperAgent Export\n")
                csv_buffer.write(f"# Export Date: {export_metadata['export_timestamp']}\n")
                csv_buffer.write(f"# Source Type: {export_metadata['source_type']}\n")
                csv_buffer.write(f"# Data Source: {export_metadata['data_source']}\n")
                if export_metadata.get('table_name'):
                    csv_buffer.write(f"# Database Table: {export_metadata['table_name']}\n")
                if export_metadata.get('custom_query'):
                    csv_buffer.write(f"# Custom Query: {export_metadata['custom_query'][:100]}...\n")
                csv_buffer.write("#\n")
            
            # Write the actual data
            export_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')
            
            filename = f"datasuper_export_{metadata.get('name', 'data').replace(' ', '_')}.csv" if metadata else "datasuper_export.csv"
            
            st.sidebar.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="export_sidebar_csv_btn"
            )
        except Exception as e:
            st.sidebar.error(f"CSV Export Error: {e}")

        # Excel Export with multiple sheets
        try:
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                # Main data sheet
                export_df.to_excel(writer, index=False, sheet_name='Data')
                
                # Metadata sheet
                if export_metadata:
                    metadata_df = pd.DataFrame([
                        {"Property": k, "Value": str(v)} for k, v in export_metadata.items()
                    ])
                    metadata_df.to_excel(writer, index=False, sheet_name='Metadata')
                
                # Analysis history sheet
                if analysis_history:
                    history_df = pd.DataFrame(analysis_history)
                    history_df.to_excel(writer, index=False, sheet_name='Analysis_History')
                
                # Schema sheet for database sources
                if metadata and metadata.get('source_type') == 'database' and 'column_types' in metadata:
                    schema_data = []
                    for col, dtype in metadata['column_types'].items():
                        schema_data.append({
                            'Column': col,
                            'Data_Type': dtype,
                            'In_Export': col in export_df.columns
                        })
                    schema_df = pd.DataFrame(schema_data)
                    schema_df.to_excel(writer, index=False, sheet_name='Schema')
            
            excel_data = output_excel.getvalue()
            filename = f"datasuper_export_{metadata.get('name', 'data').replace(' ', '_')}.xlsx" if metadata else "datasuper_export.xlsx"
            
            st.sidebar.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_sidebar_excel_btn"
            )
        except Exception as e:
            st.sidebar.error(f"Excel Export Error: {e}")

        # JSON Export with metadata
        try:
            export_json = {
                "data": export_df.to_dict(orient='records'),
            }
            
            if export_metadata:
                export_json["metadata"] = export_metadata
            
            if analysis_history:
                export_json["analysis_history"] = analysis_history
            
            json_data = pd.io.json.dumps(export_json, indent=2).encode('utf-8')
            filename = f"datasuper_export_{metadata.get('name', 'data').replace(' ', '_')}.json" if metadata else "datasuper_export.json"
            
            st.sidebar.download_button(
                label="🔗 Download as JSON",
                data=json_data,
                file_name=filename,
                mime="application/json",
                key="export_sidebar_json_btn"
            )
        except Exception as e:
            st.sidebar.error(f"JSON Export Error: {e}")
        
        # SQL Query Export for database sources
        if metadata and metadata.get('source_type') == 'database':
            try:
                sql_content = "-- DataSuperAgent SQL Export\n"
                sql_content += f"-- Export Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                sql_content += f"-- Source Database: {metadata.get('db_path', 'Unknown')}\n"
                
                if metadata.get('table_name'):
                    sql_content += f"-- Source Table: {metadata['table_name']}\n"
                    sql_content += f"\n-- Recreate table structure:\n"
                    sql_content += f"-- SELECT * FROM {metadata['table_name']};\n\n"
                
                if metadata.get('custom_query'):
                    sql_content += f"-- Original Custom Query:\n"
                    sql_content += f"{metadata['custom_query']}\n\n"
                
                # Add some useful queries
                sql_content += "-- Useful queries for this data:\n"
                sql_content += f"-- SELECT COUNT(*) FROM {metadata.get('table_name', 'your_table')};\n"
                sql_content += f"-- SELECT * FROM {metadata.get('table_name', 'your_table')} LIMIT 10;\n"
                
                if metadata.get('foreign_keys'):
                    sql_content += "\n-- Foreign Key Relationships:\n"
                    for fk in metadata['foreign_keys']:
                        sql_content += f"-- {fk['from']} -> {fk['table']}.{fk['to']}\n"
                
                filename = f"datasuper_queries_{metadata.get('name', 'data').replace(' ', '_')}.sql"
                
                st.sidebar.download_button(
                    label="🗄️ Download SQL Queries",
                    data=sql_content.encode('utf-8'),
                    file_name=filename,
                    mime="text/plain",
                    key="export_sidebar_sql_btn"
                )
            except Exception as e:
                st.sidebar.error(f"SQL Export Error: {e}")
    
    else:
        st.sidebar.caption("Load data to enable export options.")

    st.sidebar.divider()
    st.sidebar.subheader("Session Usage Stats")
    llm_calls = st.session_state.get('session_llm_calls', 0)
    estimated_tokens = st.session_state.get('session_estimated_tokens', 0.0)
    st.sidebar.write(f"LLM Calls This Session: {llm_calls}")
    st.sidebar.write(f"Estimated Tokens This Session: {int(estimated_tokens)}")
    st.sidebar.caption("Token count is a rough estimate (prompt/response length).")

    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by [Abhay Singh](https://github.com/AbhaySingh989)")
    st.sidebar.markdown("Powered by [Streamlit](https://streamlit.io), [LangChain](https://langchain.com), and [Google Gemini](https://deepmind.google/technologies/gemini/)")
