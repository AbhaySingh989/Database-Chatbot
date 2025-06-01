# DataSuper Agent 🧠✨

## 1. About the Agent

The DataSuper Agent is an interactive Streamlit web application designed for intelligent data analysis and visualization. It empowers users to upload datasets (CSV or Excel files), explore them through automated profiling, and interact with the data using a conversational AI agent powered by Google's Gemini large language models through the LangChain framework.

The agent can understand natural language queries, perform data manipulations using Pandas, generate Python code, and create visualizations with Matplotlib.

## 2. Key Features & Capabilities

*   **📁 File Upload:** Supports uploading single or multiple CSV and Excel (`.xls`, `.xlsx`) files.
    *   *Note:* Currently, if multiple files are loaded, they are automatically concatenated into a single DataFrame. More advanced merging/joining strategies are planned for future enhancements.
*   **📊 Automated Data Profiling:** After loading data, the agent displays:
    *   A preview of the data (first 5 rows).
    *   Dataset dimensions (total rows and columns).
    *   Memory usage.
    *   Column data types.
    *   Missing value counts per column.
    *   Descriptive statistics for numerical columns.
    *   Unique value counts for categorical/object columns.
*   **💬 AI-Powered Data Analysis (Chat):**
    *   Engage in a conversation with the AI agent to analyze your data.
    *   Ask questions in natural language (e.g., "What's the average sales?", "Filter for rows where region is 'North'").
    *   The agent generates and executes Python (Pandas) code to answer queries.
*   **📈 Plotting:**
    *   Request various plots (e.g., "Plot sales over time", "Show a histogram of age").
    *   The agent uses Matplotlib to generate plots, which are displayed in the chat.
*   **💡 Follow-up Suggestions:** The AI suggests relevant follow-up questions based on your interactions.
*   **🚀 Quick Analyses:** Buttons available to trigger common analyses instantly:
    *   Descriptive Statistics
    *   Missing Value Counts
    *   Correlation Matrix (with heatmap plot attempt)
*   **📤 Export Processed Data:** Download the loaded (and potentially processed by the agent, though direct agent modification of the base DataFrame is not yet a core feature) DataFrame in:
    *   CSV format
    *   Excel (.xlsx) format
    *   JSON format
*   **📜 Export Chat History:** Download the entire conversation with the agent as a text file.
*   **🪙 Estimated Token Usage:**
    *   Displays the number of LLM calls and an estimated token count for the current session in the sidebar.
    *   *Note:* Token count is a rough estimate based on prompt and response character lengths.
*   **🔄 Session Reset:** "Clear Chat History" button now fully resets the session, including loaded data, agent state, and token counts.

## 3. Setup and Running the Application

### Prerequisites
*   Python (version 3.9 or higher recommended)
*   Access to Google Gemini API (with an API key)

### Steps

1.  **Clone the Repository (if applicable):**
    If you've obtained this as a project folder, ensure you have all the files, especially `super_agent-v4.py` (or the main script file) and `requirements.txt`.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Navigate to the project directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Google API Key:**
    You need to set your Google API Key as an environment variable named `GOOGLE_API_KEY`.
    You can set it in your terminal session:
    *   Linux/macOS: `export GOOGLE_API_KEY="YOUR_API_KEY"`
    *   Windows (cmd): `set GOOGLE_API_KEY=YOUR_API_KEY`
    *   Windows (PowerShell): `$env:GOOGLE_API_KEY="YOUR_API_KEY"`
    Alternatively, you can hardcode it in the script (e.g., in `super_agent-v4.py`, around line 19), but this is **not recommended for security reasons**, especially if you share your code.

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run super_agent-v4.py
    ```
    (Replace `super_agent-v4.py` with the actual name of the main Python script if it differs).

    The application should open in your web browser automatically.

## 4. Dependencies

The core dependencies are listed in `requirements.txt`. Key libraries include:

*   `streamlit`: For creating the web application interface.
*   `pandas`: For data manipulation and analysis.
*   `langchain`, `langchain-experimental`, `langchain-google-genai`: For the AI agent framework and Google Gemini integration.
*   `matplotlib`: For generating plots.
*   `openpyxl`: For reading and writing Excel files.

## 5. Workflow Overview

1.  **Step 1: Upload Data Files:** Upload one or more CSV or Excel files.
2.  **Step 2: Load & Preview Files:** Click "Load Files for Preview".
    *   If a single file is loaded, its profile is displayed, and you can proceed to prepare the agent. Export options for this data also become available in the sidebar.
    *   If multiple files are loaded, they are currently concatenated. (The UI for individual previews and advanced merge configuration was attempted but faced tool limitations during development and is not fully active).
3.  **Step 3: Prepare AI Agent:** Click "Prepare Agent" to initialize the AI for analysis on the loaded data.
4.  **Step 4: Chat with your Data Agent:** Interact with the agent by typing questions. Use Quick Analyses buttons for common tasks.
5.  **Sidebar Options:**
    *   Clear chat history (full session reset).
    *   Download chat history.
    *   Export processed data.
    *   View status and debug info (including token usage).

---
