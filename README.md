# 🧠 DataSuperAgent Enhanced - Smart Data Analysis Assistant ✨

> 🚀 **Now with SQLite Database Support!** 

A powerful and intuitive Streamlit web application designed for interactive data analysis. Upload CSV/Excel files **OR** SQLite databases, explore table relationships, execute custom SQL queries, and engage in conversations with an AI-powered agent to analyze and visualize your data like never before!

---

## 👤 **Author**

*   **Abhay Singh** 👨‍💻
*   📧 **Email**: [abhay.rkvv@gmail.com](mailto:abhay.rkvv@gmail.com)
*   🐙 **GitHub**: [AbhaySingh989](https://github.com/AbhaySingh989)
*   💼 **LinkedIn**: [Abhay Singh](https://www.linkedin.com/in/abhay-pratap-singh-905510149/)

---

## 📖 About The Project

DataSuperAgent Enhanced is a next-generation Streamlit web application that revolutionizes data analysis workflows. Whether you're working with flat files or relational databases, this tool makes complex data exploration accessible through natural language conversations with AI.

🎯 **Perfect for**: Data analysts, researchers, business intelligence professionals, students, and anyone who wants to analyze data without writing complex code!

## ✨ Key Features

### 📁 **Multi-Format Data Support**
*   **File Formats**: CSV, Excel (`.xls`, `.xlsx`)
*   **🗄️ Database Support**: SQLite (`.db`, `.sqlite`, `.sqlite3`)
*   **🔗 Multi-Source Analysis**: Work with multiple datasets simultaneously
*   **🔄 Smart Data Combination**: Intelligently merge compatible datasets

### 🗄️ **Advanced Database Features** ⭐ NEW!
*   **📋 Table Explorer**: Browse database schemas with foreign key relationships
*   **🔍 SQL Query Interface**: Execute custom queries with syntax validation
*   **📊 Table Preview**: View sample data and column information
*   **🔐 Security First**: Built-in SQL injection prevention
*   **📜 Query History**: Save and reuse your favorite queries

### 🤖 **AI-Powered Analysis**
*   **🧠 Context-Aware AI**: Database schema and relationships inform AI responses
*   **💬 Natural Language Queries**: Ask questions in plain English
*   **📈 Smart Visualizations**: Auto-generated charts and plots
*   **💡 Intelligent Suggestions**: Context-aware follow-up recommendations
*   **🔄 Multi-Dataset Context**: AI understands your data relationships

### 📊 **Advanced Analytics**
*   **📈 Dynamic Plotting**: Line charts, bar charts, histograms, scatter plots
*   **🔗 Cross-Dataset Analysis**: Compare and combine multiple data sources
*   **📋 Schema Compatibility**: Automatic compatibility checking
*   **🎯 Relationship Mapping**: Visualize foreign key relationships

### 🛡️ **Security & Performance**
*   **🔒 SQL Injection Protection**: Enterprise-grade security validation
*   **⚡ Query Caching**: Lightning-fast repeated queries
*   **💾 Memory Optimization**: Efficient handling of large datasets
*   **🔄 Connection Pooling**: Optimized database connections

### 📤 **Enhanced Export Options**
*   **📄 Multiple Formats**: CSV, Excel, JSON, SQL scripts
*   **📋 Rich Metadata**: Include source information and analysis history
*   **🔒 Data Anonymization**: Privacy-friendly export options
*   **📊 Analysis Reports**: Export complete analysis workflows

## 🛠️ Tech Stack

DataSuperAgent Enhanced is built with cutting-edge technologies:

### 🐍 **Core Technologies**
*   **Language**: Python 3.7+
*   **Web Framework**: Streamlit
*   **Database**: SQLite3 (built-in)
*   **Data Processing**: Pandas, Openpyxl

### 🤖 **AI & Machine Learning**
*   **LangChain**: Advanced framework for LLM applications
*   **LangChain Experimental**: Pandas DataFrame agents
*   **Google Gemini 2.0-flash**: State-of-the-art language model
*   **Context-Aware Prompting**: Database schema integration

### 📊 **Visualization & UI**
*   **Matplotlib**: Publication-quality plots
*   **Plotly**: Interactive visualizations
*   **Streamlit Components**: Rich UI elements

### 🔧 **Development & Security**
*   **Python-dotenv**: Environment configuration
*   **Tabulate**: Data formatting
*   **Security Validation**: SQL injection prevention
*   **Performance Optimization**: Caching and memory management

## 🚀 Quick Start Guide

Follow these simple steps to get DataSuperAgent Enhanced running on your computer!

### 📋 Prerequisites

Before you begin, make sure you have:

1. **🐍 Python 3.7+** 
   - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check installation: `python --version`

2. **🔑 Google AI Studio API Key** (Free!)
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Click "Get API Key" → "Create API Key"
   - 📋 Copy your key (keep it safe!)

### 💻 Installation Steps

#### Step 1: Download the Project 📥
```bash
# Option A: Clone with Git
git clone https://github.com/AbhaySingh989/Database-Chatbot.git
cd Database-Chatbot

# Option B: Download ZIP from GitHub and extract
```

#### Step 2: Set Up Python Environment 🐍
```bash
# Create a virtual environment (recommended)
python -m venv datasuper_env

# Activate it
# On Windows:
datasuper_env\Scripts\activate
# On Mac/Linux:
source datasuper_env/bin/activate
```

#### Step 3: Install Dependencies 📦
```bash
# Install all required packages
pip install -r requirements.txt
```

#### Step 4: Configure API Key 🔑

**🎯 Easy Method (Recommended for beginners):**
1. Create a file named `.env` in the project folder
2. Add this line: `GOOGLE_API_KEY=your_actual_api_key_here`
3. Replace `your_actual_api_key_here` with your real API key

**Example `.env` file:**
```
GOOGLE_API_KEY=AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### Step 5: Launch the Application 🚀
```bash
# Start the application
streamlit run app.py

# Or alternatively:
python -m streamlit run app.py
```

🎉 **Success!** The app will open in your browser at `http://localhost:8501`

### 🔧 Alternative Setup Methods

<details>
<summary>📱 <strong>One-Click Setup Script</strong> (Advanced users)</summary>

Create a `setup.bat` (Windows) or `setup.sh` (Mac/Linux) file:

**Windows (`setup.bat`):**
```batch
@echo off
python -m venv datasuper_env
datasuper_env\Scripts\activate
pip install -r requirements.txt
echo.
echo Setup complete! Now create your .env file with your Google API key.
echo Then run: streamlit run app.py
pause
```

**Mac/Linux (`setup.sh`):**
```bash
#!/bin/bash
python3 -m venv datasuper_env
source datasuper_env/bin/activate
pip install -r requirements.txt
echo "Setup complete! Now create your .env file with your Google API key."
echo "Then run: streamlit run app.py"
```
</details>

<details>
<summary>🐳 <strong>Docker Setup</strong> (For developers)</summary>

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t datasuper-agent .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key_here datasuper-agent
```
</details>

## 📈 How to Use DataSuperAgent Enhanced

### 🎯 **4-Step Workflow**

#### **Step 1: Upload Your Data** 📁
Choose your data source:

**📄 For Files (CSV/Excel):**
- Click "Files (CSV/Excel)" tab
- Drag & drop or browse for your files
- Supports multiple files at once!

**🗄️ For Databases (SQLite):**
- Click "Database (SQLite)" tab  
- Upload your `.db`, `.sqlite`, or `.sqlite3` file
- 🔍 Browse tables and preview data
- 📋 View schema and relationships
- ⚡ Execute custom SQL queries

#### **Step 2: Load and Explore** 🔄
- **Files**: Click "Load and Combine Uploaded Files"
- **Databases**: Select tables or write custom queries
- 👀 Preview your data structure
- 📊 See row/column counts and data types

#### **Step 3: Prepare AI Agent** 🤖
- Click "Prepare Agent" 
- 🧠 AI learns your data structure
- 🔗 Database relationships are automatically detected
- 💡 Get personalized analysis suggestions

#### **Step 4: Start Analyzing!** 💬
Ask questions in plain English:
- *"Show me the top 10 customers by sales"*
- *"Create a chart of monthly trends"*
- *"What are the correlations between these columns?"*
- *"Find outliers in the data"*

### 🎨 **Example Conversations**

<details>
<summary>📊 <strong>Data Exploration Examples</strong></summary>

**Basic Analysis:**
```
You: "What does this data look like?"
AI: Shows data summary, types, and basic statistics

You: "Show me the first 10 rows"
AI: Displays formatted table with top rows

You: "How many records do we have?"
AI: Provides total count and breakdown by categories
```

**Visualization Requests:**
```
You: "Create a bar chart of sales by region"
AI: Generates and displays interactive bar chart

You: "Show me a correlation heatmap"
AI: Creates correlation matrix visualization

You: "Plot the trend over time"
AI: Generates time series line chart
```
</details>

<details>
<summary>🗄️ <strong>Database-Specific Examples</strong></summary>

**Schema Exploration:**
```
You: "What tables are related to customers?"
AI: Shows foreign key relationships and suggests joins

You: "Explain the database structure"
AI: Provides overview of tables, relationships, and key fields
```

**Advanced Queries:**
```
You: "Show me customers with their total order amounts"
AI: Automatically joins tables and calculates totals

You: "Find the top products by revenue this year"
AI: Writes complex query with date filtering and aggregation
```
</details>

### 🔧 **Advanced Features**

#### **🔄 Multi-Dataset Management**
- Switch between different data sources
- Compare datasets side-by-side  
- Combine compatible data sources
- Maintain separate chat histories

#### **📤 Smart Export Options**
- 📄 **CSV/Excel**: With metadata and analysis history
- 🔗 **JSON**: Structured data with full context
- 🗄️ **SQL Scripts**: Recreate your database queries
- 📊 **Analysis Reports**: Complete workflow documentation

#### **🛡️ Security & Performance**
- 🔒 **SQL Injection Protection**: Enterprise-grade security
- ⚡ **Smart Caching**: Faster repeated operations  
- 💾 **Memory Optimization**: Handle large datasets efficiently
- 📊 **Performance Monitoring**: Track system performance

## 🐛 Troubleshooting Guide

### 🔑 **API Key Issues**
**Problem**: `GOOGLE_API_KEY environment variable not set!`
**Solutions**:
- ✅ Check your `.env` file exists and contains your API key
- ✅ Verify no extra spaces around the `=` sign
- ✅ Restart the application after creating `.env`
- ✅ Try setting environment variable: `export GOOGLE_API_KEY=your_key`

### 📦 **Installation Problems**
**Problem**: `ModuleNotFoundError` or import errors
**Solutions**:
- ✅ Activate your virtual environment first
- ✅ Run `pip install -r requirements.txt` again
- ✅ Check Python version: `python --version` (needs 3.7+)
- ✅ Try: `pip install --upgrade pip` then reinstall

### 🗄️ **Database Issues**
**Problem**: "Database file is locked" or connection errors
**Solutions**:
- ✅ Close other applications using the database
- ✅ Check file permissions (needs read access)
- ✅ Try copying database to a different location
- ✅ Restart the application

### 📊 **Visualization Problems**
**Problem**: Plots not appearing or display issues
**Solutions**:
- ✅ Check browser console for JavaScript errors
- ✅ Try refreshing the page (Ctrl+F5)
- ✅ Ensure `temp_plots/` directory can be created
- ✅ Check terminal for matplotlib errors

### 🚀 **Performance Issues**
**Problem**: Slow loading or memory errors
**Solutions**:
- ✅ Use smaller datasets for testing
- ✅ Add `LIMIT` clauses to large SQL queries
- ✅ Clear cache in sidebar → Performance Monitor
- ✅ Restart application to free memory

### 💡 **Getting Help**

**🔍 Built-in Help System:**
- Click "❓ Help & Support" in the sidebar
- Check "Error Statistics" for common issues
- Use "Technical Details" in error messages

**🌐 Community Support:**
- 📧 Email: [abhay.rkvv@gmail.com](mailto:abhay.rkvv@gmail.com)
- 🐙 GitHub Issues: [Report a bug](https://github.com/AbhaySingh989/Database-Chatbot/issues)
- 💼 LinkedIn: [Connect with the author](https://www.linkedin.com/in/abhay-pratap-singh-905510149/)

## 📚 Additional Resources

### 🎓 **Learning Materials**
- 📖 [Streamlit Documentation](https://docs.streamlit.io/)
- 🤖 [LangChain Guide](https://python.langchain.com/docs/get_started/introduction)
- 🧠 [Google AI Studio](https://aistudio.google.com/app/prompts/new_chat)
- 🐍 [Pandas Tutorial](https://pandas.pydata.org/docs/user_guide/index.html)

### 🗄️ **Database Resources**
- 📊 [SQLite Tutorial](https://www.sqlitetutorial.net/)
- 🔍 [SQL Query Examples](https://www.w3schools.com/sql/)
- 📋 [Database Design Basics](https://www.lucidchart.com/pages/database-diagram/database-design)

### 🎯 **Sample Datasets**
Try these datasets to get started:
- 📈 [Sample Sales Data](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)
- 👥 [Customer Analytics](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- 🏪 [Retail Database](https://www.sqlitetutorial.net/sqlite-sample-database/)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 **Fork the repository**
2. 🌿 **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. 💾 **Commit changes**: `git commit -m 'Add amazing feature'`
4. 📤 **Push to branch**: `git push origin feature/amazing-feature`
5. 🔄 **Open a Pull Request**

### 🐛 **Bug Reports**
Found a bug? Please include:
- 🖥️ Operating system and Python version
- 📋 Steps to reproduce the issue
- 📸 Screenshots if applicable
- 📄 Error messages from terminal

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to the amazing open-source community:

- 🤖 **Google AI** - For the powerful Gemini language model
- 🔗 **LangChain Team** - For the excellent LLM framework
- 🎨 **Streamlit Team** - For making web apps accessible to Python developers
- 🐼 **Pandas Community** - For the incredible data manipulation library
- 📊 **Matplotlib/Plotly** - For beautiful data visualizations
- 🗄️ **SQLite Team** - For the reliable embedded database engine

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=AbhaySingh989/Database-Chatbot&type=Date)](https://star-history.com/#AbhaySingh989/Database-Chatbot&Date)

---

<div align="center">

**🚀 Ready to supercharge your data analysis?**

[**Get Started Now!**](#-quick-start-guide) | [**View Demo**](https://github.com/AbhaySingh989/Database-Chatbot) | [**Report Bug**](https://github.com/AbhaySingh989/Database-Chatbot/issues)

---

Made with ❤️ by [Abhay Singh](https://github.com/AbhaySingh989)

*Empowering data analysis through conversational AI* 🤖✨

</div>
