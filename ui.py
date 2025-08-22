"""
Clean UI Module for DataSuperAgent Enhanced
Created with minimal, working features that can be expanded incrementally
"""

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

TEMP_PLOT_DIR = "temp_plots"
TEMP_PLOT_FILE = os.path.abspath(os.path.join(TEMP_PLOT_DIR, "temp_plot.png"))

def get_followup_suggestions(prompt, response, df_columns, _llm):
    """Generates follow-up question suggestions using the suggestion LLM."""
    if not _llm: 
        return []
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
        prompt_tokens = _llm.get_num_tokens(suggestion_prompt)
        response_tokens = _llm.get_num_tokens(suggestions_text)
        st.session_state.session_estimated_tokens += (prompt_tokens + response_tokens)

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
    """Main UI function with clean, working implementation"""
    st.title("🧠 DataSuperAgent Enhanced - Smart Data Analysis Assistant ✨")
    st.caption("Upload CSV/Excel files or SQLite databases, and chat with an AI agent about your data.")

    # Initialize session state
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

    # Initialize data manager
    if "data_manager" not in st.session_state:
        try:
            from data_manager import DataManager
            st.session_state.data_manager = DataManager()
        except ImportError:
            st.warning("⚠️ Advanced database features not available. File upload still works!")
            st.session_state.data_manager = None

    # Step 1: Upload Data Files
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
                st.success(f"✅ Database file uploaded: {uploaded_db.name}")
                
                # Simple database loading
                with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                    tmp_file.write(uploaded_db.read())
                    temp_db_path = tmp_file.name
                
                try:
                    if st.session_state.data_manager:
                        from database_handler import SQLiteHandler
                        db_handler = SQLiteHandler()
                        is_valid, error_msg = db_handler.validate_database_file(temp_db_path)
                        
                        if is_valid:
                            db_handler.connect(temp_db_path)
                            tables = db_handler.get_tables()
                            
                            if tables:
                                st.write(f"**Found {len(tables)} tables:**")
                                table_names = [t.name for t in tables]
                                
                                selected_table = st.selectbox(
                                    "Select a table to load:",
                                    options=table_names,
                                    key="table_selector"
                                )
                                
                                if selected_table:
                                    # Show table preview
                                    with st.expander("Table Preview", expanded=False):
                                        preview_df = db_handler.get_table_preview(selected_table)
                                        st.dataframe(preview_df)
                                    
                                    if st.button("Load Selected Table", key="load_table_btn"):
                                        try:
                                            source_id = st.session_state.data_manager.add_database_source(
                                                temp_db_path, 
                                                table_name=selected_table,
                                                name=f"{uploaded_db.name}.{selected_table}"
                                            )
                                            st.success(f"✅ Table '{selected_table}' loaded successfully!")
                                            st.session_state.uploaded_files_processed = False
                                            st.session_state.combined_df = st.session_state.data_manager.get_active_dataset()
                                            
                                        except Exception as e:
                                            st.error(f"Failed to load table: {str(e)}")
                            else:
                                st.warning("No tables found in the database.")
                            
                            db_handler.disconnect()
                        else:
                            st.error(f"❌ Invalid database file: {error_msg}")
                    else:
                        st.info("📄 Database features require additional setup. Please use CSV/Excel files for now.")
                
                except Exception as e:
                    st.error(f"Error processing database: {str(e)}")
                finally:
                    try:
                        os.unlink(temp_db_path)
                    except:
                        pass

    # Step 2: Load and Combine Data
    with st.expander("Step 2: Load and Combine Data"):
        if uploaded_files and not st.session_state.uploaded_files_processed:
            if st.button("Load and Combine Uploaded Files"):
                try:
                    combined_df_result = load_and_combine_files(uploaded_files)
                    if combined_df_result is not None:
                        if isinstance(combined_df_result, list):
                            if not combined_df_result:
                                st.warning("No data was loaded from the files.")
                                st.session_state.combined_df = None
                            elif len(combined_df_result) == 1 and isinstance(combined_df_result[0], dict) and 'df' in combined_df_result[0]:
                                st.session_state.combined_df = combined_df_result[0]['df']
                            elif all(isinstance(item, dict) and 'df' in item for item in combined_df_result):
                                try:
                                    dfs_to_concat = [item['df'] for item in combined_df_result if item.get('df') is not None and not item['df'].empty]
                                    if dfs_to_concat:
                                        st.session_state.combined_df = pd.concat(dfs_to_concat, ignore_index=True)
                                        st.info(f"Multiple files combined. Total rows: {len(st.session_state.combined_df)}")
                                    else:
                                        st.warning("No valid DataFrames found to concatenate.")
                                        st.session_state.combined_df = None
                                except Exception as e:
                                    st.error(f"Error concatenating DataFrames: {str(e)}")
                                    st.session_state.combined_df = None
                            else:
                                st.error("Loaded data is in an unexpected format.")
                                st.session_state.combined_df = None
                        elif isinstance(combined_df_result, pd.DataFrame):
                            st.session_state.combined_df = combined_df_result
                        else:
                            st.error("Loaded data is not in a recognizable format.")
                            st.session_state.combined_df = None

                        st.session_state.agent = None
                        st.session_state.agent_ready = False
                        st.session_state.messages = []
                        st.session_state.uploaded_files_processed = True
                        
                        if st.session_state.combined_df is not None:
                            st.write("Preview of Combined Data (first 5 rows):")
                            st.dataframe(st.session_state.combined_df.head())
                    else:
                        st.error("Failed to load or combine data.")
                        st.session_state.combined_df = None
                        st.session_state.uploaded_files_processed = False
                        
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")
                    st.session_state.combined_df = None

        elif st.session_state.combined_df is not None:
             st.success("Data is loaded and ready for analysis.")
             st.write("Preview of Active Data (first 5 rows):")
             st.dataframe(st.session_state.combined_df.head())
        else:
             st.info("Upload files or databases in Step 1 and load them here.")

    # Step 3: Prepare AI Agent
    with st.expander("Step 3: Prepare AI Agent for Analysis"):
        if st.session_state.combined_df is not None and not st.session_state.agent_ready:
            if st.button("Prepare Agent"):
                try:
                    llm_agent_model = get_llm(model_name="gemini-2.0-flash", temperature=0)
                    st.session_state.llm_agent_model = llm_agent_model
                    st.session_state.agent = create_agent(st.session_state.combined_df, llm_agent_model)
                    if st.session_state.agent is not None:
                        st.session_state.agent_ready = True
                        st.success("✅ AI Agent is ready! Proceed to Step 4 to chat.")
                    else:
                        st.error("Agent preparation failed. Check logs.")
                        st.session_state.agent_ready = False
                except Exception as e:
                    st.error(f"Error preparing agent: {str(e)}")
                    st.session_state.agent_ready = False

        elif st.session_state.agent_ready:
             st.success("✅ AI Agent is ready! Proceed to Step 4 to chat.")
        else:
             st.info("Load and combine data in Step 2 first.")

    # Step 4: Chat Interface
    st.divider()
    st.subheader("Step 4: Chat with your Data Agent")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("plot_path") and os.path.exists(message["plot_path"]):
                 st.image(message["plot_path"])
            elif message.get("plot_path"):
                 st.caption("[Plot image not found - may have been cleared]")

    # Handle chat input
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
                    
                    # Clear previous plot
                    if os.path.exists(TEMP_PLOT_FILE):
                        try:
                            os.remove(TEMP_PLOT_FILE)
                        except Exception as e:
                            print(f"Warning: Could not remove previous plot file: {e}")

                    try:
                        print(f"\n>>> Running Agent with Input:\n{prompt}\n")
                        response = st.session_state.agent.invoke({"input": f"{prompt}"})
                        final_answer = response.get("output") if isinstance(response, dict) else response
                        print(f"\n<<< Agent Raw Response:\n{final_answer}\n")

                        st.session_state.session_llm_calls += 1
                        prompt_tokens = st.session_state.llm_agent_model.get_num_tokens(prompt)
                        response_tokens = st.session_state.llm_agent_model.get_num_tokens(final_answer)
                        st.session_state.session_estimated_tokens += (prompt_tokens + response_tokens)

                        time.sleep(0.5)
                        if os.path.exists(TEMP_PLOT_FILE):
                            plot_path_this_turn = TEMP_PLOT_FILE
                            message_placeholder.markdown(final_answer)
                            st.image(plot_path_this_turn)
                        else:
                            message_placeholder.markdown(final_answer)

                        # Generate follow-up suggestions
                        suggestions = []
                        if "error executing code" not in final_answer.lower() and st.session_state.combined_df is not None:
                            llm_suggestion_model = get_llm(model_name="gemini-2.0-flash", temperature=0.7)
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
                        error_message = f"😞 Agent Execution Failed: {str(e)}"
                        print("\n--- Agent Execution Error ---")
                        print(traceback.format_exc())
                        print("-----------------------------")
                        st.error(error_message)
                        final_answer = error_message
                        
                        if os.path.exists(TEMP_PLOT_FILE):
                             try: 
                                 os.remove(TEMP_PLOT_FILE)
                             except Exception as e_rem: 
                                 print(f"Error removing plot file after exception: {e_rem}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "plot_path": plot_path_this_turn
            })

def sidebar_ui():
    """Clean sidebar UI with essential features"""
    st.sidebar.header("Options")
    
    # Clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.current_prompt = ""
        st.session_state.combined_df = None
        st.session_state.agent = None
        st.session_state.agent_ready = False
        st.session_state.uploaded_files_processed = False
        st.session_state.session_llm_calls = 0
        st.session_state.session_estimated_tokens = 0.0

        if os.path.exists(TEMP_PLOT_FILE):
           try:
               os.remove(TEMP_PLOT_FILE)
           except Exception as e:
               print(f"Warning: Could not remove plot file: {e}")

        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Status & Debug Info")
    st.sidebar.write(f"Agent Ready: {'✅ Yes' if st.session_state.agent_ready else '❌ No'}")

    # Show data info
    if 'combined_df' in st.session_state and st.session_state.combined_df is not None:
        st.sidebar.write("Combined DataFrame Info:")
        if isinstance(st.session_state.combined_df, pd.DataFrame):
            st.sidebar.write(f"- Rows: {len(st.session_state.combined_df)}")
            st.sidebar.write(f"- Columns: {len(st.session_state.combined_df.columns)}")
            with st.sidebar.expander("Show Column Names"):
                st.code(st.session_state.combined_df.columns.tolist())
            with st.sidebar.expander("Show Head (First 5 Rows)"):
                st.dataframe(st.session_state.combined_df.head())
        else:
            st.sidebar.warning("Data is not in expected DataFrame format.")
    else:
        st.sidebar.warning("No data loaded/combined yet.")

    # Export functionality
    st.sidebar.divider()
    st.sidebar.subheader("📤 Export Data")

    if 'combined_df' in st.session_state and isinstance(st.session_state.combined_df, pd.DataFrame) and not st.session_state.combined_df.empty:
        try:
            csv_data = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name="datasuper_export.csv",
                mime="text/csv",
                key="export_csv_btn"
            )
        except Exception as e:
            st.sidebar.error(f"CSV Export Error: {e}")

        try:
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                st.session_state.combined_df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_data = output_excel.getvalue()
            st.sidebar.download_button(
                label="Download Data as Excel",
                data=excel_data,
                file_name="datasuper_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_excel_btn"
            )
        except Exception as e:
            st.sidebar.error(f"Excel Export Error: {e}")

        try:
            json_data = st.session_state.combined_df.to_json(orient='records', indent=4).encode('utf-8')
            st.sidebar.download_button(
                label="Download Data as JSON",
                data=json_data,
                file_name="datasuper_export.json",
                mime="application/json",
                key="export_json_btn"
            )
        except Exception as e:
            st.sidebar.error(f"JSON Export Error: {e}")
    else:
        st.sidebar.caption("Load data to enable export options.")

    # Usage stats
    st.sidebar.divider()
    st.sidebar.subheader("Session Usage Stats")
    llm_calls = st.session_state.get('session_llm_calls', 0)
    estimated_tokens = st.session_state.get('session_estimated_tokens', 0.0)
    st.sidebar.write(f"LLM Calls This Session: {llm_calls}")
    st.sidebar.write(f"Estimated Tokens This Session: {int(estimated_tokens)}")
    st.sidebar.caption("Token count is a rough estimate.")

    # Credits
    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by [Abhay Singh](https://github.com/AbhaySingh989)")
    st.sidebar.markdown("Powered by [Streamlit](https://streamlit.io), [LangChain](https://langchain.com), and [Google Gemini](https://deepmind.google/technologies/gemini/)")