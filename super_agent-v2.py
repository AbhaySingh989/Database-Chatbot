import streamlit as st
import pandas as pd
import os
import re
import time
import traceback
import io # For handling byte streams from uploads

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

# --- Data Loading and Combination ---
@st.cache_data(show_spinner="Loading and combining files...")
def load_and_combine_files(uploaded_files):
    """Loads data from uploaded files (CSV, Excel), combines them, returns DataFrame or None."""
    if not uploaded_files:
        return None

    all_dfs = []
    file_details = [] # To track which files were processed

    for file in uploaded_files:
        file_name = file.name
        file_type = file.type
        file_size = file.size

        try:
            df = None
            # Make sure to reset the file pointer before reading
            file.seek(0)
            if file_name.endswith('.csv'):
                df = pd.read_csv(file)
                file_details.append({"name": file_name, "type": "CSV", "rows": len(df)})
            elif file_name.endswith(('.xls', '.xlsx')):
                # Make sure you installed 'openpyxl': pip install openpyxl
                df = pd.read_excel(file, engine='openpyxl')
                file_details.append({"name": file_name, "type": "Excel", "rows": len(df)})
            # elif file_name.endswith('.pdf'):
            #     # PDF processing is complex - requires libraries like PyPDF2, pdfplumber, etc.
            #     # Placeholder - add actual PDF table extraction logic here if needed later
            #     st.warning(f"PDF processing for '{file_name}' is not implemented in this version.")
            #     file_details.append({"name": file_name, "type": "PDF", "status": "Skipped"})
            else:
                 st.warning(f"Unsupported file type: {file_name}. Skipping.")
                 file_details.append({"name": file_name, "type": "Unsupported", "status": "Skipped"})

            if df is not None:
                all_dfs.append(df)

        except Exception as e:
            st.error(f"Error reading file '{file_name}': {e}")
            file_details.append({"name": file_name, "status": f"Error: {e}"})
            # Decide if one error should stop the whole process or just skip the file
            # return None # Option: Stop if any file fails

    # Display summary of files processed
    st.subheader("File Loading Summary")
    st.dataframe(pd.DataFrame(file_details))

    if not all_dfs:
        st.warning("No valid dataframes were loaded.")
        return None

    # Attempt to combine the collected DataFrames
    if len(all_dfs) == 1:
        st.success("Loaded 1 file.")
        return all_dfs[0]
    else:
        try:
            # Use concat which handles slightly different columns by default (fills with NaN)
            combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
            st.success(f"Successfully combined {len(all_dfs)} file(s). Total rows: {len(combined_df)}")
            # Optional: Display columns to check if combination makes sense
            # st.write("Columns in combined data:", combined_df.columns.tolist())
            return combined_df
        except Exception as e:
            st.error(f"Error combining DataFrames: {e}")
            st.warning("Combining failed. This might happen if files have very different structures. Proceeding with only the first loaded DataFrame.")
            # Fallback: return the first df if combination fails? Or return None?
            return all_dfs[0] # Return first as fallback

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
        Your goal is to answer questions accurately by executing Python code using the provided tools.

        **Core Requirements:**
        1.  **EXECUTE CODE:** For any request needing data calculation, filtering, aggregation, or statistics, you MUST generate and execute the relevant Python Pandas code. Do NOT provide answers without executing code.
        2.  **ACCURATE RESULTS:** Base your final answer directly on the results of the executed code.
        3.  **MARKDOWN TABLES:** When the result is a DataFrame, format the **entire** result as a standard Markdown table in your final answer. If it's long (>15 rows), show the first 15 and state that it's truncated.
        4.  **PLOTTING:**
            - If asked to plot: Generate the Python code using pandas plotting (`df.plot()`) or `matplotlib.pyplot`. **If the user specifies a plot type (e.g., 'scatter plot', 'bar chart', 'histogram'), try to use that specific type.**
            - **You MUST include `import matplotlib.pyplot as plt` in your plotting code.**
            - **Save the plot to the filename '{plot_filename}'.** Use exactly this filename. Use `plt.savefig('{plot_filename}')`.
            - **After saving, you MUST include `plt.close()` to close the plot.**
            - In your final text answer, simply confirm that the plot was generated and saved (e.g., "I have generated the requested plot and saved it."). Do not describe the plot in detail unless asked.
        5.  **ERRORS:** If code execution fails, clearly state "Error executing code:" followed by the Python error message in your final answer. Do not make up results.

        Begin! Remember to save plots to '{plot_filename}' and then call plt.close().
        """.format(plot_filename=TEMP_PLOT_FILE) # Pass the filename

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
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "current_prompt" not in st.session_state:
     st.session_state.current_prompt = ""
if "uploaded_files_processed" not in st.session_state: # Track if combine button was clicked for current uploads
    st.session_state.uploaded_files_processed = False


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
                 st.session_state.combined_df = combined_df_result
                 st.session_state.agent = None # Reset agent if data changes
                 st.session_state.agent_ready = False
                 st.session_state.messages = [] # Clear chat on new data
                 st.session_state.uploaded_files_processed = True # Mark as processed
                 st.write("Preview of Combined Data (first 5 rows):")
                 st.dataframe(st.session_state.combined_df.head())

                 # --- Automated Data Profile ---
                 st.subheader("Automated Data Profile:")
                 st.write(f"Total Rows: {st.session_state.combined_df.shape[0]}")
                 st.write(f"Total Columns: {st.session_state.combined_df.shape[1]}")

                 st.write("Memory Usage:")
                 with io.StringIO() as buffer:
                     st.session_state.combined_df.info(buf=buffer)
                     s = buffer.getvalue()
                 st.text(s)

                 st.write("Column Data Types:")
                 st.dataframe(st.session_state.combined_df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

                 st.write("Missing Values per Column:")
                 missing_values = st.session_state.combined_df.isnull().sum().reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'})
                 missing_values = missing_values[missing_values['Missing Count'] > 0]
                 if not missing_values.empty:
                     st.dataframe(missing_values)
                 else:
                     st.write("No missing values found in any column.")

                 st.write("Descriptive Statistics (Numerical Columns):")
                 # Ensure import numpy as np if not already present
                 try:
                     st.dataframe(st.session_state.combined_df.describe(include=np.number))
                 except NameError:
                     import numpy as np # Attempt to import if not found
                     st.dataframe(st.session_state.combined_df.describe(include=np.number))


                 st.subheader("Unique Value Counts (Categorical/Object Columns)")
                 # Select only object or category columns for nunique, to avoid issues with datetime etc.
                 categorical_cols = st.session_state.combined_df.select_dtypes(include=['object', 'category']).columns
                 if not categorical_cols.empty:
                     unique_counts = st.session_state.combined_df[categorical_cols].nunique().reset_index().rename(columns={'index': 'Column', 0: 'Unique Values'})
                     st.dataframe(unique_counts)
                 else:
                     st.write("No categorical/object columns found to display unique counts for.")
                 # --- End Automated Data Profile ---

             else:
                 st.error("Failed to load or combine data.")
                 st.session_state.combined_df = None # Ensure it's None if failed
                 st.session_state.uploaded_files_processed = False

    elif st.session_state.combined_df is not None:
         st.success("Data is loaded and combined.")
         st.write("Preview of Combined Data (first 5 rows):")
         st.dataframe(st.session_state.combined_df.head())

         # --- Automated Data Profile (also show if data is already loaded) ---
         st.subheader("Automated Data Profile:")
         st.write(f"Total Rows: {st.session_state.combined_df.shape[0]}")
         st.write(f"Total Columns: {st.session_state.combined_df.shape[1]}")

         st.write("Memory Usage:")
         with io.StringIO() as buffer:
             st.session_state.combined_df.info(buf=buffer)
             s = buffer.getvalue()
         st.text(s)

         st.write("Column Data Types:")
         st.dataframe(st.session_state.combined_df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

         st.write("Missing Values per Column:")
         missing_values = st.session_state.combined_df.isnull().sum().reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'})
         missing_values = missing_values[missing_values['Missing Count'] > 0]
         if not missing_values.empty:
             st.dataframe(missing_values)
         else:
             st.write("No missing values found in any column.")

         st.write("Descriptive Statistics (Numerical Columns):")
         # Ensure import numpy as np if not already present
         try:
             st.dataframe(st.session_state.combined_df.describe(include=np.number))
         except NameError:
             import numpy as np # Attempt to import if not found
             st.dataframe(st.session_state.combined_df.describe(include=np.number))

         st.subheader("Unique Value Counts (Categorical/Object Columns)")
         # Select only object or category columns for nunique, to avoid issues with datetime etc.
         categorical_cols = st.session_state.combined_df.select_dtypes(include=['object', 'category']).columns
         if not categorical_cols.empty:
             unique_counts = st.session_state.combined_df[categorical_cols].nunique().reset_index().rename(columns={'index': 'Column', 0: 'Unique Values'})
             st.dataframe(unique_counts)
         else:
             st.write("No categorical/object columns found to display unique counts for.")
         # --- End Automated Data Profile ---
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
# --- Quick Analyses Buttons ---
st.markdown("---") # Optional separator
st.markdown("**Quick Analyses:**")

QUICK_ANALYSES = [
    ("Descriptive Stats", "Show descriptive statistics for all numerical columns."),
    ("Missing Values", "Show missing value counts for all columns."),
    ("Correlation Matrix", "Calculate and show the correlation matrix for all numerical columns. Also, plot it as a heatmap if possible, saving it to the standard plot filename.")
]

# Ensure agent is ready before showing quick analysis buttons that require it
if st.session_state.agent_ready:
    num_quick_analyses = len(QUICK_ANALYSES)
    cols = st.columns(num_quick_analyses)
    for i, (label, prompt_text) in enumerate(QUICK_ANALYSES):
        button_key = f"quick_analysis_{i}"
        if cols[i].button(label, key=button_key, use_container_width=True):
            st.session_state.current_prompt = prompt_text
            st.rerun()
else:
    st.info("Prepare the agent in Step 3 to enable Quick Analyses.")

st.markdown("---") # Optional separator

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display plot if a path was stored AND the file exists
        if message.get("plot_path") and os.path.exists(message["plot_path"]):
             st.image(message["plot_path"])
        elif message.get("plot_path"):
             st.caption("[Plot image not found - may have been cleared]")


# Handle Pending Prompt (from suggestion click)
if st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    st.session_state.current_prompt = "" # Clear it
else:
    prompt = st.chat_input("Ask the agent about the loaded data...")

# Main Interaction Logic
if prompt:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt, "plot_path": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if agent is ready
    if st.session_state.agent is None or not st.session_state.agent_ready:
        st.warning("⚠️ Agent is not ready. Please complete Step 3 first.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Agent is not ready. Please prepare the agent in Step 3.",
            "plot_path": None
        })
    else:
        # Run agent
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking and executing...")
            plot_path_this_turn = None # Reset

            # Clean up previous plot file BEFORE running agent
            if os.path.exists(TEMP_PLOT_FILE):
                try:
                    os.remove(TEMP_PLOT_FILE)
                    print(f"Cleared previous plot file: {TEMP_PLOT_FILE}")
                except Exception as e_rem:
                    print(f"Warning: Could not remove previous plot file: {e_rem}")

            try:
                # === Run the Agent ===
                print(f"\n>>> Running Agent with Input:\n{prompt}\n")
                # Use invoke for potentially richer output if needed, run is simpler for text
                # response = st.session_state.agent.invoke({"input": prompt}) # invoke returns dict
                # final_answer = response.get("output", "Agent did not return 'output'.")

                response = st.session_state.agent.run(prompt) # run returns string directly
                final_answer = response
                print(f"\n<<< Agent Raw Response:\n{final_answer}\n")
                # =====================

                # --- Check for Plot File AFTER Agent Run ---
                time.sleep(0.5) # Give filesystem a moment
                if os.path.exists(TEMP_PLOT_FILE):
                    print(f"Confirmed plot file exists after agent run: {TEMP_PLOT_FILE}")
                    plot_path_this_turn = TEMP_PLOT_FILE
                    message_placeholder.markdown(final_answer) # Display text first
                    st.image(plot_path_this_turn) # Display plot image below
                else:
                    print(f"Plot file NOT found after agent run at: {TEMP_PLOT_FILE}")
                    message_placeholder.markdown(final_answer) # Display only text

                # --- Generate Follow-up Suggestions ---
                suggestions = []
                if "error executing code" not in final_answer.lower() and st.session_state.combined_df is not None:
                    suggestions = get_followup_suggestions(
                        prompt, final_answer,
                        st.session_state.combined_df.columns.tolist(),
                        llm_suggestion_model # Pass the suggestion LLM
                    )

                # Display suggestions if any
                if suggestions:
                    st.markdown("---")
                    st.markdown("**Suggested follow-up questions:**")
                    cols = st.columns(len(suggestions))
                    for i, sugg in enumerate(suggestions):
                        # Ensure unique keys for buttons within the loop and across reruns
                        button_key = f"suggestion_{len(st.session_state.messages)}_{i}"
                        if cols[i].button(sugg, key=button_key, use_container_width=True):
                            st.session_state.current_prompt = sugg
                            st.rerun() # Rerun app immediately with the new prompt

            except Exception as e:
                error_message = f"😞 Agent Execution Failed: {str(e)}"
                print("\n--- Agent Execution/Display Traceback ---")
                print(traceback.format_exc())
                print("---------------------------------------")
                st.error(error_message) # Show error in main area
                final_answer = error_message
                suggestions = []
                # Clean up plot file if error occurred after potential creation
                if os.path.exists(TEMP_PLOT_FILE):
                     try: os.remove(TEMP_PLOT_FILE)
                     except Exception as e_rem_err: print(f"Error removing plot file after exception: {e_rem_err}")

        # Append assistant's full response (text + plot path) to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "plot_path": plot_path_this_turn
        })
        # Need to rerun *if suggestions were NOT clicked* to clear the prompt input visually
        # However, Streamlit handles input clearing automatically on enter usually.
        # If a suggestion button WAS clicked, rerun() already happened.


# --- Sidebar for Options/Debug ---
st.sidebar.header("Options")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.current_prompt = "" # Clear pending prompt too
    # Clean up plot file
    if os.path.exists(TEMP_PLOT_FILE):
        try: os.remove(TEMP_PLOT_FILE)
        except Exception as e_rem: print(f"Error removing plot file on clear: {e_rem}")
    st.rerun()

# --- Export Chat History ---
def format_chat_history_for_download(messages):
    history_str = ""
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        history_str += f"{role}: {content}
"
        if message.get("plot_path") and os.path.exists(message["plot_path"]):
            history_str += f"   (Plot generated and displayed: {message['plot_path']})
"
        elif message.get("plot_path"):
            history_str += f"   (Plot path noted: {message['plot_path']}, but file may no longer be available)
"
        history_str += "
" # Add a blank line between messages
    return history_str

if st.session_state.messages: # Only show button if there's history
    chat_export_data = format_chat_history_for_download(st.session_state.messages)
    st.sidebar.download_button(
        label="Download Chat History",
        data=chat_export_data,
        file_name="datasuper_chat_history.txt",
        mime="text/plain"
    )
else:
    st.sidebar.caption("No chat history to download yet.")
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