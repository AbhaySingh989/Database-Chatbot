import traceback
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
from langchain_experimental.tools import PythonREPLTool


def create_database_context_prompt(metadata: Dict[str, Any]) -> str:
    """Create context prompt for database-aware agent"""
    context_parts = []
    
    if metadata.get('source_type') == 'database':
        context_parts.append("🗄️ **DATABASE CONTEXT:**")
        
        if 'db_path' in metadata:
            db_name = metadata['db_path'].split('/')[-1] if '/' in metadata['db_path'] else metadata['db_path']
            context_parts.append(f"- Database: {db_name}")
        
        if 'table_name' in metadata:
            context_parts.append(f"- Source Table: {metadata['table_name']}")
            
        if 'custom_query' in metadata:
            context_parts.append("- Source: Custom SQL Query")
            context_parts.append(f"- Query: {metadata['custom_query'][:100]}...")
        
        # Schema information
        if 'columns' in metadata and 'column_types' in metadata:
            context_parts.append("\n📋 **SCHEMA INFORMATION:**")
            for col in metadata['columns'][:10]:  # Limit to first 10 columns
                col_type = metadata['column_types'].get(col, 'Unknown')
                context_parts.append(f"- {col}: {col_type}")
            
            if len(metadata['columns']) > 10:
                context_parts.append(f"- ... and {len(metadata['columns']) - 10} more columns")
        
        # Relationships
        if 'foreign_keys' in metadata and metadata['foreign_keys']:
            context_parts.append("\n🔗 **RELATIONSHIPS:**")
            for fk in metadata['foreign_keys'][:5]:  # Limit to first 5 FKs
                context_parts.append(f"- {fk['from']} → {fk['table']}.{fk['to']}")
        
        # Indexes
        if 'indexes' in metadata and metadata['indexes']:
            context_parts.append(f"\n📇 **INDEXES:** {', '.join(metadata['indexes'][:3])}")
    
    elif metadata.get('source_type') == 'file':
        context_parts.append("📄 **FILE CONTEXT:**")
        if 'file_path' in metadata:
            file_name = metadata['file_path'].split('/')[-1] if '/' in metadata['file_path'] else metadata['file_path']
            context_parts.append(f"- Source File: {file_name}")
        if 'file_type' in metadata:
            context_parts.append(f"- File Type: {metadata['file_type'].upper()}")
    
    # General data info
    if 'row_count' in metadata:
        context_parts.append(f"- Total Rows: {metadata['row_count']:,}")
    if 'column_count' in metadata:
        context_parts.append(f"- Total Columns: {metadata['column_count']}")
    
    return "\n".join(context_parts)


def create_enhanced_system_prompt(metadata: Optional[Dict[str, Any]] = None) -> str:
    """Create enhanced system prompt with database context"""
    
    base_prompt = """You are a helpful data analysis assistant. You have access to a pandas DataFrame called 'df' that contains the user's data.

IMPORTANT INSTRUCTIONS:
1. Always use the variable name 'df' to refer to the DataFrame
2. When creating visualizations, save plots to 'temp_plots/temp_plot.png' using matplotlib
3. Provide clear, actionable insights about the data
4. If asked about relationships between data, consider the schema and foreign key information provided
5. When suggesting follow-up analyses, consider the data structure and relationships
6. Always explain your analysis approach and findings clearly
7. Use appropriate statistical methods and visualizations for the data type

IMPORTANT NOTE ON PYTHON_REPL TOOL:
- Each call to the 'python_repl' tool runs in a fresh, isolated Python environment.
- You MUST include all necessary 'import' statements (e.g., 'import pandas as pd') at the beginning of EACH 'Action Input' code block.
- Variables defined in one 'Action Input' will NOT be available in subsequent 'Action Input' calls.
- Therefore, ensure each 'Action Input' is a complete, self-contained script to achieve the desired outcome.

"""
    
    if metadata:
        context_info = create_database_context_prompt(metadata)
        if context_info:
            base_prompt += f"\n{context_info}\n"
            
            # Add database-specific instructions
            if metadata.get('source_type') == 'database':
                base_prompt += """
DATABASE-SPECIFIC GUIDANCE:
- Consider table relationships when analyzing data patterns
- If foreign keys exist, you can suggest analyses that would benefit from joining related tables
- Be aware of the original table structure when interpreting column meanings
- When discussing data quality, consider database constraints and relationships
- If this data came from a custom query, be mindful that it may be a subset or transformation of the original table

"""
    
    base_prompt += """
TOOLS:
You have access to the following tools:

python_repl: A Python shell. Use this to execute Python commands. Input should be a valid Python command.

To use a tool, use the following format:

Thought: I need to use the python_repl tool to execute some Python code.
Action: python_repl
Action Input: print(df.head())
Observation: ...

Remember: Focus on providing valuable insights and actionable recommendations based on the data structure and content.
"""
    
    return base_prompt


def create_agent(_df, _llm, metadata: Optional[Dict[str, Any]] = None):
    """Creates an enhanced LangChain Pandas DataFrame agent with database context awareness."""
    if _df is None or _llm is None:
        print("🚨 Cannot create agent: DataFrame or LLM is missing.")
        return None
    
    try:
        # Create enhanced system prompt with context
        system_prompt = create_enhanced_system_prompt(metadata)
        
        # Create an instance of PythonREPLTool
        python_repl_tool = PythonREPLTool()

        # Create agent with enhanced context and explicit tool
        agent = create_pandas_dataframe_agent(
            llm=_llm,
            df=_df,
            verbose=True,
            agent_type="zero-shot-react-description",
            allow_dangerous_code=True,
            prefix=system_prompt,
            tools=[python_repl_tool] # Explicitly pass the tool
        )
        
        # Log context information
        if metadata:
            print(f"✅ Enhanced Pandas DataFrame Agent created with {metadata.get('source_type', 'unknown')} context.")
            if metadata.get('source_type') == 'database':
                if 'table_name' in metadata:
                    print(f"   📋 Table: {metadata['table_name']}")
                if 'foreign_keys' in metadata and metadata['foreign_keys']:
                    print(f"   🔗 Foreign Keys: {len(metadata['foreign_keys'])}")
                if 'custom_query' in metadata:
                    print(f"   🔍 Custom Query Source")
        else:
            print(f"✅ Standard Pandas DataFrame Agent created.")
        
        return agent
        
    except Exception as e:
        print(f"🚨 Agent Creation Failed: {e}")
        print("--- Agent Creation Traceback ---")
        print(traceback.format_exc())
        return None


def get_database_suggestions(metadata: Dict[str, Any]) -> List[str]:
    """Generate database-specific analysis suggestions"""
    suggestions = []
    
    if metadata.get('source_type') != 'database':
        return suggestions
    
    # Basic analysis suggestions
    suggestions.extend([
        "Show me a summary of the data structure and key statistics",
        "What are the data quality issues I should be aware of?",
        "Create visualizations to explore the main patterns in this data"
    ])
    
    # Foreign key relationship suggestions
    if 'foreign_keys' in metadata and metadata['foreign_keys']:
        suggestions.extend([
            "Explain the relationships between this table and other tables",
            "What insights can we gain from the foreign key relationships?",
            "Are there any data integrity issues with the foreign key references?"
        ])
    
    # Column-specific suggestions based on data types
    if 'column_types' in metadata:
        numeric_cols = [col for col, dtype in metadata['column_types'].items() 
                       if 'INT' in dtype.upper() or 'REAL' in dtype.upper() or 'NUMERIC' in dtype.upper()]
        text_cols = [col for col, dtype in metadata['column_types'].items() 
                    if 'TEXT' in dtype.upper() or 'VARCHAR' in dtype.upper()]
        date_cols = [col for col, dtype in metadata['column_types'].items() 
                    if 'DATE' in dtype.upper() or 'TIME' in dtype.upper()]
        
        if numeric_cols:
            suggestions.append(f"Analyze the distribution and correlations of numeric columns: {', '.join(numeric_cols[:3])}")
        
        if text_cols:
            suggestions.append(f"Explore the categorical data in text columns: {', '.join(text_cols[:3])}")
        
        if date_cols:
            suggestions.append(f"Analyze trends over time using date columns: {', '.join(date_cols[:2])}")
    
    # Table-specific suggestions
    if 'table_name' in metadata:
        table_name = metadata['table_name'].lower()
        if 'user' in table_name or 'customer' in table_name:
            suggestions.append("Analyze user/customer demographics and behavior patterns")
        elif 'order' in table_name or 'transaction' in table_name:
            suggestions.append("Analyze transaction patterns and trends over time")
        elif 'product' in table_name or 'item' in table_name:
            suggestions.append("Explore product performance and characteristics")
    
    return suggestions[:6]  # Limit to 6 suggestions


def create_context_aware_followup_prompt(original_prompt: str, response: str, 
                                       metadata: Optional[Dict[str, Any]] = None) -> str:
    """Create context-aware follow-up prompt for suggestions"""
    
    base_prompt = f"""
Given the user's question: "{original_prompt}"
And the analysis response: "{response[:300]}..."

Suggest 3 relevant follow-up questions that would provide deeper insights.
"""
    
    if metadata and metadata.get('source_type') == 'database':
        base_prompt += """
Consider the database context when suggesting follow-ups:
- Table relationships and foreign keys
- Data types and constraints
- Potential for cross-table analysis
- Database-specific patterns and anomalies
"""
        
        if 'foreign_keys' in metadata and metadata['foreign_keys']:
            base_prompt += f"\nAvailable relationships: {[fk['table'] for fk in metadata['foreign_keys'][:3]]}"
    
    base_prompt += """
Format as:
1. Question 1?
2. Question 2?
3. Question 3?
"""
    
    return base_prompt
