# 🧠 DataSuperAgent: Production Ready - Smart Data Analysis Assistant ✨

> 🚀 **Now with Live Database Support (PostgreSQL, BigQuery, Snowflake), Automated Data Profiling, and Interactive Dashboards!**

A powerful, production-ready Streamlit web application designed for interactive data analysis. Upload files, connect to live databases, and engage in conversations with an AI-powered agent to analyze and visualize your data like never before.

---

## 📖 About The Project

DataSuperAgent has been transformed from an MVP to a robust, production-ready application. It provides a seamless interface for data analysts, data scientists, and business professionals to connect to various data sources, generate deep insights, and build interactive dashboards. This tool makes complex data exploration accessible through a powerful, AI-driven conversational interface.

🎯 **Perfect for**: Data analysts, researchers, business intelligence professionals, students, and anyone who wants to analyze data without writing complex code!

## ✨ Key Features

### 📁 **Versatile Data Connectivity**
*   **File Formats**: CSV, Excel (`.xls`, `.xlsx`), and local SQLite databases (`.db`, `.sqlite`).
*   **Live Databases**: Connect directly to **PostgreSQL**, **Google BigQuery**, and **Snowflake**.
*   **Secure Connection Forms**: Securely input credentials for live databases, which are never exposed in the UI.

### 🤖 **Advanced AI & Analytics**
*   **Automated Data Profiling**: With one click, generate a comprehensive data profile report to understand data quality, types, and statistics.
*   **Proactive AI Suggestions**: The agent suggests insightful analytical questions based on your data's schema, helping you discover hidden patterns.
*   **Context-Aware AI**: The agent understands your database schema and relationships for more accurate and relevant responses.
*   **Natural Language Queries**: Ask questions in plain English and get back analysis and visualizations.

### 📊 **Interactive Dashboarding**
*   **One-Click Dashboard Generation**: Let the AI generate a dashboard with key metrics and charts based on your data.
*   **Customizable Layout**: Freely drag, resize, and rearrange dashboard components to create a personalized view.
*   **Delete Components**: Easily remove charts or metrics from your dashboard.

## 🛠️ Tech Stack

DataSuperAgent is built with a modern, robust tech stack:

### 🐍 **Core Technologies**
*   **Language**: Python 3.9+
*   **Web Framework**: Streamlit
*   **Data Processing**: Pandas, Openpyxl

### 🗄️ **Database Connectors**
*   **SQLite**: Built-in support for file-based SQLite.
*   **PostgreSQL**: `psycopg2-binary`
*   **Google BigQuery**: `google-cloud-bigquery`
*   **Snowflake**: `snowflake-connector-python`

### 🤖 **AI & Machine Learning**
*   **LangChain**: Advanced framework for building LLM applications.
*   **Google Gemini**: Powered by the `gemini-2.0-flash` model.

### 📊 **Visualization & UI**
*   **ydata-profiling**: For automated data profile generation.
*   **streamlit-elements**: For interactive, draggable dashboard grids.
*   **Plotly & Matplotlib**: For generating a wide range of charts.

### 🧪 **Testing & CI/CD**
*   **Testing Framework**: `pytest` and `pytest-mock`.
*   **CI/CD**: Automated testing pipeline with GitHub Actions.

---

## 🚀 How to Use DataSuperAgent

This guide will walk you through using the application's powerful features.

### **Step 1: Choose Your Data Source**
The application supports two main ways to load your data, selectable from the "Step 1" expander:

#### **A) File Upload**
This is for working with local files.
1.  **CSV/Excel**: Select the "Files (CSV/Excel)" tab and upload one or more files.
2.  **SQLite**: Select the "Database (SQLite File)" tab and upload your `.db` or `.sqlite` file.

#### **B) Database Connection**
This is for connecting to live, remote databases.
1.  **Select Database Type**: Choose between PostgreSQL, Google BigQuery, or Snowflake.
2.  **Enter Credentials**:
    *   **PostgreSQL**: Fill in the secure form with your database host, port, name, user, and password.
    *   **Google BigQuery**: Upload your Service Account JSON file.
    *   **Snowflake**: Fill in the form with your user, password, account, warehouse, database, and schema information.
3.  **Connect**: Click the "Connect" button. The application will attempt to connect and list the available tables.

### **Step 2: Load Your Data**
Once you have selected a data source (e.g., uploaded a file or selected a table from a database connection), proceed to "Step 2" and click the **"Load Data"** button. This will load your selected data into the application's memory.

### **Step 3: Prepare the AI Agent**
With your data loaded, go to "Step 3" and click the **"Prepare Agent"** button. This initializes the AI with the context of your data's schema.

After the agent is prepared, it will proactively suggest some analytical questions you can ask. You can click on any of these buttons to start your analysis.

### **Step 4: Analyze Your Data**
This is where the magic happens! You have three tabs for analysis:

#### **💬 Chat with Agent**
*   Simply type your question into the chat input at the bottom and press Enter.
*   The agent will analyze the data and respond with text, tables, and sometimes charts.

#### **📊 View Data Profile**
*   Click this tab to see a comprehensive, automated report of your dataset.
*   It includes details on data types, missing values, correlations, and more. This is a great way to perform an initial Exploratory Data Analysis (EDA).

#### **📈 Dashboard**
*   Click the **"✨ Generate Dashboard"** button. The AI will analyze your data and create a dashboard of relevant metrics and charts.
*   **Customize**: You can drag the charts to rearrange them, resize them from their bottom-right corner, and delete any chart by clicking the "❌" icon at its top.

---

## 🛠️ Local Installation Guide

### **Prerequisites**
1.  **Python 3.9+**
2.  **Google AI Studio API Key**: Get one from [aistudio.google.com](https://aistudio.google.com/).

### **Installation Steps**
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AbhaySingh989/Database-Chatbot.git
    cd Database-Chatbot
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Mac/Linux
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure your API Key:**
    *   Create a file named `.env` in the project root.
    *   Add the following line, replacing `your_key_here` with your actual key:
        ```
        GOOGLE_API_KEY=your_key_here
        ```
5.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```
The application should now be open in your web browser!

---
*Original project author: Abhay Singh. This version has been significantly refactored and enhanced.*
