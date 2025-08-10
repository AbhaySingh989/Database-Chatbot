# 🧠 DataSuperAgent - Smart Data Analysis Assistant ✨

A powerful and intuitive Streamlit web application designed for interactive data analysis. It empowers users to upload their datasets, combine them seamlessly, and then engage in a conversation with an AI-powered agent to explore, analyze, and visualize the data.

---

👤 **Author**

*   **Abhay Singh**
*   📧 **Email**: [abhay.rkvv@gmail.com](mailto:abhay.rkvv@gmail.com)
*   🐙 **GitHub**: [AbhaySingh989](https://github.com/AbhaySingh989)
*   💼 **LinkedIn**: [Abhay Singh](https://www.linkedin.com/in/abhay-pratap-singh-905510149/)

---

## 📖 About The Project

DataSuperAgent is a Streamlit-based web application built to simplify the data analysis workflow. It allows users to directly interact with their data through a conversational AI, making complex data queries and visualizations accessible without needing to write code manually.

At its core, the agent utilizes Google's Gemini language model, accessed via the Langchain framework, to translate natural language questions into executable Python code for data analysis.

## ✨ Features

*   **📁 Intuitive File Handling:**
    *   Easily upload data in popular formats like CSV and Excel (`.xls`, `.xlsx`).
    *   Intelligently combines multiple uploaded files into a single dataset for comprehensive analysis.
*   **🤖 AI-Powered Analysis:**
    *   Leverage the power of Google's Gemini LLM through Langchain to query your data using natural language.
    *   The agent generates and executes Python (Pandas) code to perform data operations.
*   **📊 Dynamic Plotting:**
    *   Request various plots and charts (e.g., line charts, bar charts, histograms).
    *   The agent generates these using Matplotlib and displays them directly in the chat interface.
*   **💡 Intelligent Suggestions:**
    *   To facilitate deeper exploration, the agent provides relevant follow-up questions based on the current analysis.
*   **⚙️ User-Friendly Interface:**
    *   Built with Streamlit for a clean and interactive user experience.
    *   Efficient caching for data and AI models to improve performance.
    *   Keeps track of your conversation with the agent.

## 🛠️ Tech Stack

DataSuperAgent is built with a modern stack of Python libraries and AI technologies:

*   **Core Language:** Python
*   **Web Framework:** Streamlit
*   **Data Handling:** Pandas
*   **AI & Machine Learning:**
    *   **Langchain:** Framework for developing applications powered by language models.
    *   **Google Gemini:** The core Large Language Model (`gemini-2.0-flash`).
*   **Plotting:** Matplotlib
*   **File Parsing:** Openpyxl

## 🚀 Getting Started

This guide will walk you through setting up and running the DataSuperAgent on your local machine.

### Prerequisites

1.  **Python:** Version 3.7 or higher. You can download it from [python.org](https://www.python.org/downloads/).
2.  **Google AI Studio API Key:** The agent uses Google's Gemini Pro LLM.
    *   Go to [Google AI Studio](https://aistudio.google.com/).
    *   Create a new API key and copy it.

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone https://github.com/AbhaySingh989/Database-Chatbot.git
    cd Database-Chatbot
    ```

2.  **Set up your Google API Key:**
    *   **Option A: Directly Edit the Script (Recommended for beginners)**
        1.  Open `super_agent-v2.py`.
        2.  Find the line: `os.environ["GOOGLE_API_KEY"] = "..."`
        3.  Replace the placeholder with your actual Google API key.
    *   **Option B: Use an Environment Variable (More secure)**
        Set an environment variable named `GOOGLE_API_KEY` with your key.

3.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```bash
    python -m streamlit run super_agent-v2.py
    ```
    The application should open in your browser at `http://localhost:8501`.

## 📈 How It Works

The application follows a simple, multi-step workflow:

1.  **Step 1: Upload Data:** Upload one or more CSV or Excel files.
2.  **Step 2: Load and Combine:** Click the "Load and Combine" button to process the files into a single DataFrame.
3.  **Step 3: Prepare AI Agent:** Click "Prepare Agent" to initialize the AI with your data.
4.  **Step 4: Chat:** Ask questions in natural language to analyze and visualize your data!

## 🐛 Troubleshooting

*   **`GOOGLE_API_KEY` Error:** Ensure your API key is set correctly, either in the script or as an environment variable.
*   **`ModuleNotFoundError`:** Make sure you have installed all dependencies from `requirements.txt`.
*   **Plots Not Appearing:** Check the terminal for errors. Ensure the `temp_plots` directory can be created and written to.

## 🙏 Acknowledgments

*   Google AI for the powerful Gemini model.
*   The LangChain team for their excellent framework.
*   The Streamlit team for making web app creation in Python so accessible.

---
Made with ❤️ by Abhay Singh
