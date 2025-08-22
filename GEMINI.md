# GEMINI.md

## Project Overview

This project, "DataSuperAgent Enhanced," is a powerful and intuitive Streamlit web application designed for interactive data analysis. It allows users to upload CSV/Excel files or SQLite databases, explore table relationships, execute custom SQL queries, and engage in conversations with an AI-powered agent to analyze and visualize data.

The application is built with Python and leverages several key technologies:

*   **Web Framework:** Streamlit
*   **AI and Machine Learning:** LangChain, LangChain Experimental, and Google Gemini
*   **Data Processing:** Pandas, Openpyxl
*   **Database:** SQLite3

## Building and Running

To build and run this project, follow these steps:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Up Environment Variables:**
    Create a `.env` file in the root of the project and add your Google API key:
    ```
    GOOGLE_API_KEY=your_google_api_key
    ```

3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Development Conventions

*   **Modular Structure:** The code is organized into modules with specific responsibilities (e.g., `database_handler.py`, `agent_handler.py`, `ui.py`).
*   **Object-Oriented Programming:** The `database_handler.py` module uses classes (`SQLiteHandler`) and data classes (`TableInfo`, `ColumnInfo`, `QueryResult`) to model database interactions.
*   **Error Handling:** The application includes error handling to gracefully manage issues like missing API keys and application errors.
*   **Security and Performance:** The `security_performance.py` module provides functionality for SQL injection prevention, query caching, and memory optimization.
*   **Virtual Environment:** The project uses a virtual environment (`datasuper_env`) to manage dependencies.
