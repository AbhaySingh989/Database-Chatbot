import streamlit as st
import pandas as pd
import os
import re
import time
import traceback
import io # For handling byte streams from uploads
from itertools import combinations # For find_potential_common_columns

# Langchain & Google specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Plotting specific setup
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Data Agent", layout="wide", page_icon="🧠")

# --- Configuration ---
# Define a fixed path for the temporary plot file the agent will create
TEMP_PLOT_DIR = "temp_plots" # Store plots in a sub-directory
TEMP_PLOT_FILE = os.path.abspath(os.path.join(TEMP_PLOT_DIR, "temp_plot.png"))
# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDN9tpQVBIPlm8gP3IdI3OWLY2eqv5EiDY"  # Replace with your actual API key

# Create the temp plot directory if it doesn't exist
os.makedirs(TEMP_PLOT_DIR, exist_ok=True)

# --- API Key Handling (Check Environment Variable) ---
if "GOOGLE_API_KEY" not in os.environ:
    st.error("🚨 GOOGLE_API_KEY environment variable not set!")
    st.info("Please set the GOOGLE_API_KEY environment variable before running.")
    st.stop()
else:
    # Optionally display a masked key or confirmation (DO NOT print the full key)
    # st.sidebar.success("GOOGLE_API_KEY found.")
    pass

# --- LLM Initialization ---
# Cache LLM instances using Streamlit's resource caching
@st.cache_resource
def get_llm(model_name, temperature):
    """Creates and returns a cached ChatGoogleGenerativeAI instance."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        print(f"LLM '{model_name}' initialized.")
        return llm
    except Exception as e:
        st.error(f"🚨 Failed to initialize LLM '{model_name}': {e}")
        print(f"--- LLM Init Traceback ({model_name}) ---")
        print(traceback.format_exc())
        print("-----------------------------------------")
        st.stop() # Stop the app if LLM fails

# Get the LLM instances (will be created only once)
# --- LLM Initialization ---
# Define model names explicitly
AGENT_MODEL_NAME = "gemini-2.0-flash"
SUGGESTION_MODEL_NAME = "gemini-2.0-flash" # Or choose a different one if desired

# Cache LLM instances using Streamlit's resource caching
@st.cache_resource
def get_llm(model_name, temperature):
    """Creates and returns a cached ChatGoogleGenerativeAI instance."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        print(f"LLM '{model_name}' initialized.")
        return llm
    except Exception as e:
        st.error(f"🚨 Failed to initialize LLM '{model_name}': {e}")
        print(f"--- LLM Init Traceback ({model_name}) ---")
        print(traceback.format_exc())
        print("-----------------------------------------")
        st.stop() # Stop the app if LLM fails


# Get the LLM instances using the defined names
llm_agent_model = get_llm(model_name=AGENT_MODEL_NAME, temperature=0)
llm_suggestion_model = get_llm(model_name=SUGGESTION_MODEL_NAME, temperature=0.7)

@st.cache_data(show_spinner="Loading files...") # Modified spinner message
def load_and_combine_files(uploaded_files):
    """
    Loads data from uploaded files (CSV, Excel).
    Returns a list of dictionaries, each containing 'name' and 'df', or None if no files.
    """
    if not uploaded_files:
        return None

    loaded_dfs_info = []
    file_details_summary = [] # For the summary table

    for file in uploaded_files:
        file_name = file.name
        # file_type = file.type # Not strictly needed for the new logic but kept for now
        # file_size = file.size # Not strictly needed for the new logic but kept for now

        try:
            df = None
            file.seek(0) # Reset file pointer
            if file_name.endswith('.csv'):
                df = pd.read_csv(file)
                file_details_summary.append({"name": file_name, "type": "CSV", "rows": len(df)})
            elif file_name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file, engine='openpyxl')
                file_details_summary.append({"name": file_name, "type": "Excel", "rows": len(df)})
            else:
                 st.warning(f"Unsupported file type: {file_name}. Skipping.")
                 file_details_summary.append({"name": file_name, "type": "Unsupported", "status": "Skipped"})

            if df is not None:
                loaded_dfs_info.append({'name': file_name, 'df': df})

        except Exception as e:
            st.error(f"Error reading file '{file_name}': {e}")
            file_details_summary.append({"name": file_name, "status": f"Error: {e}"})
            # Continue to try and load other files

    # Display summary of files processed
    st.subheader("File Loading Summary")
    st.dataframe(pd.DataFrame(file_details_summary))

    if not loaded_dfs_info:
        st.warning("No valid dataframes were loaded.")
        return None

    return loaded_dfs_info

# --- Function to find potential common columns ---
def find_potential_common_columns(list_of_df_infos):
    """
    Finds columns with exact matching names between all unique pairs of DataFrames.
    Input: list_of_df_infos (e.g., st.session_state.individual_dfs)
    Output: Dictionary {('df_name1', 'df_name2'): ['common_col1', ...], ...}
    """
    if not list_of_df_infos or len(list_of_df_infos) < 2:
        return {}

    common_columns_candidates = {}
    # from itertools import combinations # Already imported globally

    for (info1, info2) in combinations(list_of_df_infos, 2):
        df1_name = info1['name']
        df1 = info1['df']
        df2_name = info2['name']
        df2 = info2['df']

        common_cols = list(set(df1.columns) & set(df2.columns))
        if common_cols:
            common_columns_candidates[(df1_name, df2_name)] = common_cols
    return common_columns_candidates

# --- Agent Creation ---
@st.cache_resource(show_spinner="Initializing Data Analyst Agent...")
def create_agent(_df, _llm):
    """Creates and caches a LangChain Pandas DataFrame agent."""
    if _df is None or _llm is None:
        st.error("🚨 Cannot create agent: DataFrame or LLM is missing.")
        return None
    try:
        agent_instructions_prefix = """
        You are an expert data analyst working with a Pandas DataFrame named `df`.
        Your goal is to answer questions accurately and efficiently by executing Python code.

        **Core Requirements:**
        1.  **EXECUTE CODE:** For any request needing data calculation, filtering, aggregation, or statistics, you MUST generate and execute the relevant Python Pandas code. Do NOT provide answers without executing code.
        2.  **ACCURATE RESULTS:** Base your final answer directly on the results of the executed code.
        3.  **MARKDOWN TABLES:** When the result is a DataFrame, format the **entire** result as a standard Markdown table in your final answer. If it's long (>15 rows), show the first 15 and state that it's truncated.
        4.  **PLOTTING:**
            - If asked to plot: Generate Python code using `pandas.DataFrame.plot()` or `matplotlib.pyplot`. If the user specifies a plot type (e.g., 'scatter plot', 'bar chart', 'histogram'), try to use that specific type.
            - **You MUST include `import matplotlib.pyplot as plt` in your plotting code block.**
            - **Save the plot to the filename '{plot_filename}'.** Use exactly this filename and path. Use `plt.savefig('{plot_filename}')`.
            - **After saving, you MUST include `plt.close()` to close the plot and free up memory.**
            - In your final text answer, simply confirm plot generation and saving (e.g., "I have generated the requested plot."). Do not describe the plot unless asked.
        5.  **CODE QUALITY & EFFICIENCY:**
            - Write clean, efficient, and idiomatic Python Pandas code.
            - Aim for correctness and completeness in your first attempt to minimize iterations.
            - Use `pandas` for data manipulation and `numpy` for numerical operations effectively.
        6.  **CONTEXTUAL AWARENESS:** If the user's current query seems to build upon previous interactions in this session, utilize the chat history to maintain context.
        7.  **CONCISE COMMUNICATION:** Provide brief explanations for your actions. Be concise unless the user asks for more detail.
        8.  **ERRORS:** If code execution fails, clearly state "Error executing code:" followed by the Python error message in your final answer. Do not make up results.

        Remember to save plots to '{plot_filename}' and then call `plt.close()`.
        Begin!
        """.format(plot_filename=TEMP_PLOT_FILE)

        agent = create_pandas_dataframe_agent(
            llm=_llm,
            df=_df,
            prefix=agent_instructions_prefix,
            verbose=True, # Set to False for cleaner Streamlit output, True for terminal debugging
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            agent_executor_kwargs={'handle_parsing_errors': True},
        )
        print(f"Pandas DataFrame Agent created successfully. Target plot filename: {TEMP_PLOT_FILE}")
        return agent

    except Exception as e:
        st.error(f"🚨 Agent Creation Failed: {e}")
        print("--- Agent Creation Traceback ---")
        print(traceback.format_exc())
        print("------------------------------")
        return None

# --- Suggestion Generation ---
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

        # Token estimation for suggestion call
        if 'session_llm_calls' not in st.session_state: # Ensure init
            st.session_state.session_llm_calls = 0
        if 'session_estimated_tokens' not in st.session_state:
            st.session_state.session_estimated_tokens = 0.0

        st.session_state.session_llm_calls += 1
        # `suggestion_prompt` is the input to the suggestion LLM
        # `suggestions_text` is the LLM's string output
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

# --- Streamlit App UI ---


st.title("🧠 Smart Data Analysis Agent ✨")
st.caption(f"Upload CSV/Excel files, combine them, and chat with an AI agent (using `{AGENT_MODEL_NAME}`) about the data.")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "combined_df" not in st.session_state:
    st.session_state.combined_df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "individual_dfs" not in st.session_state: # New state for list of DF infos
    st.session_state.individual_dfs = []
if "potential_merge_candidates" not in st.session_state: # New state for merge candidates
    st.session_state.potential_merge_candidates = {}
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "current_prompt" not in st.session_state:
     st.session_state.current_prompt = ""
if "uploaded_files_processed" not in st.session_state: # Track if combine button was clicked for current uploads
    st.session_state.uploaded_files_processed = False
if 'session_llm_calls' not in st.session_state:
    st.session_state.session_llm_calls = 0
if 'session_estimated_tokens' not in st.session_state:
    st.session_state.session_estimated_tokens = 0.0


# --- Workflow Steps ---

# Expander 1: File Upload
with st.expander("Step 1: Upload Data Files (CSV or Excel)", expanded=True):
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls'], # Specify allowed types
        key="file_uploader"
    )
    if uploaded_files:
        st.session_state.uploaded_files_processed = False # New files uploaded, reset flag

# Expander 2: Load and Combine Data
with st.expander("Step 2: Load and Combine Data"):
    if uploaded_files and not st.session_state.uploaded_files_processed:
        if st.button("Load and Combine Uploaded Files"):
             combined_df_result = load_and_combine_files(uploaded_files)
             if combined_df_result is not None:
                 st.session_state.combined_df = combined_df_result # This line assumes load_and_combine_files returns a single DF
                 st.session_state.agent = None
                 st.session_state.agent_ready = False
                 st.session_state.messages = []
                 st.session_state.uploaded_files_processed = True
                 st.write("Preview of Combined Data (first 5 rows):")
                 if isinstance(st.session_state.combined_df, list):
                     if st.session_state.combined_df and \
                        isinstance(st.session_state.combined_df[0], dict) and \
                        'df' in st.session_state.combined_df[0] and \
                        isinstance(st.session_state.combined_df[0]['df'], pd.DataFrame):
                         st.dataframe(st.session_state.combined_df[0]['df'].head())
                         # Optionally, provide context if it's from a list:
                         # st.caption("Displaying preview of the first DataFrame from the loaded list.")
                     else:
                         # Handle cases where it's a list but not in the expected format
                         st.warning("Data loaded in an unexpected list format. Cannot display head preview here.")
                 elif isinstance(st.session_state.combined_df, pd.DataFrame):
                     st.dataframe(st.session_state.combined_df.head())
                 else:
                     st.warning("Combined data is not a recognizable DataFrame or list of DataFrames. Cannot display preview.")
             else:
                 st.error("Failed to load or combine data.")
                 st.session_state.combined_df = None
                 st.session_state.uploaded_files_processed = False

    elif st.session_state.combined_df is not None: # This is the block for when data is already loaded and is a DataFrame
         st.success("Data is loaded and combined.")
         st.write("Preview of Combined Data (first 5 rows):")
         # This part should be safe as combined_df is confirmed to be a DataFrame here by prior logic
         if isinstance(st.session_state.combined_df, pd.DataFrame):
             st.dataframe(st.session_state.combined_df.head())
         else:
             # This case should ideally not be reached if session state is managed correctly,
             # but added for robustness if combined_df was somehow set to a non-DataFrame type.
             st.warning("Previously loaded data is not in the expected DataFrame format. Cannot display preview.")
    else:
         st.info("Upload files in Step 1 and click 'Load and Combine'.")

# Expander 3: Prepare AI Agent
with st.expander("Step 3: Prepare AI Agent for Analysis"):
    if st.session_state.combined_df is not None and not st.session_state.agent_ready:
        if st.button("Prepare Agent"):
            st.session_state.agent = create_agent(st.session_state.combined_df, llm_agent_model)
            if st.session_state.agent is not None:
                st.session_state.agent_ready = True
                st.success("✅ AI Agent is ready! Proceed to Step 4 to chat.")
            else:
                st.error("Agent preparation failed. Check logs.")
                st.session_state.agent_ready = False

    elif st.session_state.agent_ready:
         st.success("✅ AI Agent is ready! Proceed to Step 4 to chat.")
    else:
         st.info("Load and combine data in Step 2 first.")


# --- Step 4: Chat with Agent ---
st.divider()
st.subheader("Step 4: Chat with your Data Agent")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("plot_path") and os.path.exists(message["plot_path"]):
             st.image(message["plot_path"])
        elif message.get("plot_path"):
             st.caption("[Plot image not found - may have been cleared]")


# Handle Pending Prompt (from suggestion click)
if st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    st.session_state.current_prompt = ""
else:
    prompt = st.chat_input("Ask the agent about the loaded data...")

# Main Interaction Logic
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
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking and executing...")
            plot_path_this_turn = None
            if os.path.exists(TEMP_PLOT_FILE):
                try:
                    os.remove(TEMP_PLOT_FILE)
                    print(f"Cleared previous plot file: {TEMP_PLOT_FILE}")
                except Exception as e_rem:
                    print(f"Warning: Could not remove previous plot file: {e_rem}")

            try:
                print(f"\n>>> Running Agent with Input:\n{prompt}\n")
                response = st.session_state.agent.run(prompt)
                final_answer = response
                print(f"\n<<< Agent Raw Response:\n{final_answer}\n")
                # Token estimation for agent call
                if 'session_llm_calls' not in st.session_state: # Ensure init if somehow missed
                    st.session_state.session_llm_calls = 0
                if 'session_estimated_tokens' not in st.session_state:
                    st.session_state.session_estimated_tokens = 0.0

                st.session_state.session_llm_calls += 1
                # `prompt` is the user's input to the agent
                # `final_answer` is the agent's string output
                prompt_len = len(str(prompt if prompt is not None else ""))
                response_len = len(str(final_answer if final_answer is not None else ""))
                st.session_state.session_estimated_tokens += (prompt_len + response_len) / 4.0
                # =====================

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
                print("\n--- Agent Execution/Display Traceback ---")
                print(traceback.format_exc())
                print("---------------------------------------")
                st.error(error_message)
                final_answer = error_message
                suggestions = []
                if os.path.exists(TEMP_PLOT_FILE):
                     try: os.remove(TEMP_PLOT_FILE)
                     except Exception as e_rem_err: print(f"Error removing plot file after exception: {e_rem_err}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "plot_path": plot_path_this_turn
        })

# --- Sidebar for Options/Debug ---
st.sidebar.header("Options")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.current_prompt = ""
    st.session_state.combined_df = None
    st.session_state.agent = None # Crucial for agent re-initialization
    st.session_state.agent_ready = False
    st.session_state.uploaded_files_processed = False # Reset flag for Step 2 logic

    # Variables from attempted merge refactoring (reset for cleanliness)
    st.session_state.individual_dfs = []
    st.session_state.potential_merge_candidates = {}
    # If 'llm_merge_suggestions' was added to session_state, reset it too:
    if 'llm_merge_suggestions' in st.session_state:
       st.session_state.llm_merge_suggestions = None # Or {} depending on its typical type

    if 'uploaded_file_names' in st.session_state: # Reset this as well
        st.session_state.uploaded_file_names = []

    # Reset token count variables
    st.session_state.session_llm_calls = 0
    st.session_state.session_estimated_tokens = 0.0

    # Clean up plot file (existing logic)
    if os.path.exists(TEMP_PLOT_FILE):
       try:
           os.remove(TEMP_PLOT_FILE)
           print(f"Cleared plot file on session reset: {TEMP_PLOT_FILE}")
       except Exception as e_rem:
           print(f"Warning: Could not remove plot file on session reset: {e_rem}")

    st.rerun() # Existing logic

st.sidebar.divider()
st.sidebar.header("Status & Debug Info")
st.sidebar.write(f"Agent Ready: {'✅ Yes' if st.session_state.agent_ready else '❌ No'}")
if st.session_state.combined_df is not None:
     st.sidebar.write("Combined DataFrame Info:")
     st.sidebar.write(f"- Rows: {len(st.session_state.combined_df)}")
     st.sidebar.write(f"- Columns: {len(st.session_state.combined_df.columns)}")
     with st.sidebar.expander("Show Column Names"):
        st.code(st.session_state.combined_df.columns.tolist())
     with st.sidebar.expander("Show Head (First 5 Rows)"):
        st.dataframe(st.session_state.combined_df.head())
else:
     st.sidebar.warning("No data loaded/combined yet.")

st.sidebar.divider()
st.sidebar.subheader("Export Processed Data")

if 'combined_df' in st.session_state and isinstance(st.session_state.combined_df, pd.DataFrame) and not st.session_state.combined_df.empty:
    try:
        csv_data = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="datasuper_export.csv",
            mime="text/csv",
            key="export_sidebar_csv_btn"
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
            key="export_sidebar_excel_btn"
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
            key="export_sidebar_json_btn"
        )
    except Exception as e:
        st.sidebar.error(f"JSON Export Error: {e}")
else:
    st.sidebar.caption("Load data (Step 2) to enable export options.")

st.sidebar.divider()
st.sidebar.subheader("Session Usage Stats")
# Ensure keys exist before accessing
llm_calls = st.session_state.get('session_llm_calls', 0)
estimated_tokens = st.session_state.get('session_estimated_tokens', 0.0)
st.sidebar.write(f"LLM Calls This Session: {llm_calls}")
st.sidebar.write(f"Estimated Tokens This Session: {int(estimated_tokens)}")
st.sidebar.caption("Token count is a rough estimate (prompt/response length).")