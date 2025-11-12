import traceback
import re
import json
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class LangGraphAgent:
    def __init__(self, llm, df):
        self.llm = llm
        self.df = df
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        graph.add_edge("tools", "agent")
        return graph.compile()

    def _should_continue(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        if "tool_calls" not in last_message.additional_kwargs:
            return "end"
        else:
            return "continue"

    def _agent_node(self, state):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _tools_node(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        tool_call = last_message.additional_kwargs["tool_calls"][0]
        tool = self._get_tool(tool_call["function"]["name"])
        response = tool.invoke(tool_call["function"]["arguments"])
        return {"messages": [response]}

    def _get_tool(self, name):
        if name == "python_repl":
            return PythonAstREPLTool(locals={"df": self.df})
        else:
            raise ValueError(f"Unknown tool: {name}")

    def invoke(self, query):
        return self.graph.invoke({"messages": [HumanMessage(content=query)]})


def create_chat_agent(_df, _llm, metadata: Optional[Dict[str, Any]] = None):
    """Creates a conversational agent for Q&A."""
    if _df is None or _llm is None:
        print("🚨 Cannot create chat agent: DataFrame or LLM is missing.")
        return None

    try:
        # Create a specialized prompt for the chat agent
        chat_prompt = f"""
        You are a data analysis expert. You are talking to a user who has provided you with a pandas DataFrame.
        The user will ask you questions about the data. You have a tool that can execute Python code against the dataframe.
        Your task is to answer the user's questions by using the Python REPL tool to analyze the data.

        Here is the context of the data you are working with:
        {create_database_context_prompt(metadata)}

        Here is the head of the dataframe:
        {_df.head().to_markdown()}

        When you are asked a question, you should first think about what you need to do to answer the question.
        Then, you should use the `python_repl` tool to execute the necessary Python code.
        Finally, you should use the output of the tool to answer the user's question.

        You MUST use the following format:

        Thought: Do I need to use a tool? Yes
        Action: The action to take. Should be one of [python_repl]
        Action Input: The input to the action
        Observation: The result of the action

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        """
        # Create the LangGraph agent
        agent = LangGraphAgent(llm=_llm, df=_df)
        
        # Log context information
        if metadata:
            print(f"✅ Chat Agent created with {metadata.get('source_type', 'unknown')} context.")
        else:
            print(f"✅ Standard Chat Agent created.")

        return agent

    except Exception as e:
        print(f"🚨 Chat Agent Creation Failed: {e}")
        print("--- Chat Agent Creation Traceback ---")
        print(traceback.format_exc())
        return None


def create_dashboarding_agent(_df, _llm, metadata: Optional[Dict[str, Any]] = None):
    """Creates a specialized agent for generating dashboard definitions."""
    if _df is None or _llm is None:
        print("🚨 Cannot create dashboarding agent: DataFrame or LLM is missing.")
        return None

    try:
        # Create a specialized prompt for dashboarding
        columns_str = ", ".join(metadata.get('columns', []))
        dashboard_prompt = f"""
        You are a data visualization expert. Your task is to generate a dashboard definition in JSON format for a 12-column grid.
        Based on the table schema with columns: [{columns_str}], define a dashboard that provides meaningful insights into the data.

        Choose the best chart types from this list: ['metric', 'bar_chart', 'line_chart', 'table'].
        For each chart, you MUST define a grid layout with 'x', 'y', 'w', and 'h' integer properties.
        Each chart object in the JSON response must have unique keys.
        'x' is the horizontal position (0-11). 'y' is the vertical position. 'w' is width, 'h' is height.
        - 'metric': For KPIs. Should be small (e.g., w=3, h=1). Requires 'column' and 'expression' ('SUM', 'AVERAGE', 'COUNT').
        - 'bar_chart': For categories. Requires 'x_axis' and 'y_axis' columns.
        - 'line_chart': For trends. Requires 'x_axis' and 'y_axis' columns.
        - 'table': For raw data. Requires a list of 'columns'.

        Return ONLY a valid JSON object which is a list of chart definitions. Do not include any explanation.

        Example JSON response:
        ```json
        [
          {{
            "type": "metric", "label": "Total Sales", "column": "sales", "expression": "SUM",
            "x": 0, "y": 0, "w": 3, "h": 1
          }},
          {{
            "type": "bar_chart", "label": "Sales by Region", "x_axis": "region", "y_axis": "sales",
            "x": 3, "y": 0, "w": 9, "h": 2
          }}
        ]
        ```
        """
        
        response = _llm.invoke(dashboard_prompt)
        content = response.content if hasattr(response, 'content') else response
        print("LLM response for dashboard definition:")
        print(content)
        
        # Clean the response to extract only the JSON part
        # Look for a JSON block enclosed in ```json ... ```
        match = re.search(r'```json\s*(\[.*\])\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback to finding the first list
            json_str_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_str_match:
                json_str = json_str_match.group(0)
            else:
                json_str = ""

        if json_str:
            dashboard_def = json.loads(json_str)
            print("Parsed dashboard definition:")
            print(dashboard_def)
            return dashboard_def
        return []

    except Exception as e:
        print(f"🚨 Dashboarding Agent Creation Failed: {e}")
        print("--- Dashboarding Agent Creation Traceback ---")
        print(traceback.format_exc())
        return None


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


def get_database_suggestions(metadata: Dict[str, Any]) -> List[str]:
    """Generate database-specific analysis suggestions"""
    suggestions = []
    
    if metadata.get('source_type') != 'database':
        return suggestions
    
    # Prioritize the most specific suggestions first.
    
    # Table-specific suggestions
    if 'table_name' in metadata:
        table_name = metadata['table_name'].lower()
        if 'user' in table_name or 'customer' in table_name:
            suggestions.append("Analyze user/customer demographics and behavior patterns")
        elif 'order' in table_name or 'transaction' in table_name:
            suggestions.append("Analyze transaction patterns and trends over time")
        elif 'product' in table_name or 'item' in table_name:
            suggestions.append("Explore product performance and characteristics")

    # Foreign key relationship suggestions
    if 'foreign_keys' in metadata and metadata['foreign_keys']:
        suggestions.extend([
            "Explain the relationships between this table and other tables",
            "What insights can we gain from the foreign key relationships?",
        ])

    # Column-specific suggestions based on data types
    if 'column_types' in metadata:
        numeric_cols = [col for col, dtype in metadata['column_types'].items()
                       if 'INT' in dtype.upper() or 'REAL' in dtype.upper() or 'NUMERIC' in dtype.upper()]
        text_cols = [col for col, dtype in metadata['column_types'].items()
                    if 'TEXT' in dtype.upper() or 'VARCHAR' in dtype.upper()]
        
        if numeric_cols:
            suggestions.append(f"Analyze the distribution and correlations of numeric columns: {', '.join(numeric_cols[:3])}")
        
        if text_cols:
            suggestions.append(f"Explore the categorical data in text columns: {', '.join(text_cols[:3])}")

    # Basic analysis suggestions (add them last as they are most generic)
    if len(suggestions) < 6:
        suggestions.extend([
            "Show me a summary of the data structure and key statistics",
            "What are the data quality issues I should be aware of?",
            "Create visualizations to explore the main patterns in this data"
        ])

    return suggestions[:6]  # Limit to 6 suggestions



def get_proactive_suggestions(metadata: Dict[str, Any], llm) -> List[str]:
    """
    Uses an LLM to generate proactive analytical questions based on the data schema.
    """
    if not llm or not metadata or 'columns' not in metadata:
        return []

    columns_str = ", ".join(metadata.get('columns', []))
    prompt = f"""
    Based on the following table schema with columns: [{columns_str}],
    generate 3 to 5 insightful analytical questions a business analyst might ask.
    The questions should be concise and directly answerable from the data.
    Return ONLY the questions as a numbered list, each on a new line.

    Example:
    1. What is the trend of sales over time?
    2. Which product category has the highest revenue?
    """

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else response
        
        # Parse the numbered list response
        suggestions = re.findall(r"^\s*\d+\.\s+(.*)", content, re.MULTILINE)
        return [s.strip() for s in suggestions]
    except Exception as e:
        print(f"Error generating proactive suggestions: {e}")
        return []


def generate_dashboard_definition(metadata: Dict[str, Any], llm) -> List[Dict]:
    """
    Uses an LLM to generate a dashboard definition from the data schema.
    """
    if not llm or not metadata or 'columns' not in metadata:
        return []

    columns_str = ", ".join(metadata.get('columns', []))
    prompt = f"""
    You are a data analyst creating a dashboard definition in JSON format for a 12-column grid.
    Based on the table schema with columns: [{columns_str}], define a dashboard.
    
    Choose the best chart types from this list: ['metric', 'bar_chart', 'line_chart', 'table'].
    For each chart, you MUST define a grid layout with 'x', 'y', 'w', and 'h' integer properties.
    Each chart object in the JSON response must have unique keys.
    'x' is the horizontal position (0-11). 'y' is the vertical position. 'w' is width, 'h' is height.
    - 'metric': For KPIs. Should be small (e.g., w=3, h=1). Requires 'column' and 'expression' ('SUM', 'AVERAGE', 'COUNT').
    - 'bar_chart': For categories. Requires 'x_axis' and 'y_axis' columns.
    - 'line_chart': For trends. Requires 'x_axis' and 'y_axis' columns.
    - 'table': For raw data. Requires a list of 'columns'.
    
    Return ONLY a valid JSON object which is a list of chart definitions. Do not include any explanation.

    Example JSON response:
    ```json
    [
      {{
        "type": "metric", "label": "Total Sales", "column": "sales", "expression": "SUM",
        "x": 0, "y": 0, "w": 3, "h": 1
      }},
      {{
        "type": "bar_chart", "label": "Sales by Region", "x_axis": "region", "y_axis": "sales",
        "x": 3, "y": 0, "w": 9, "h": 2
      }}
    ]
    ```
    """

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else response
        print("LLM response for dashboard definition:")
        print(content)
        
        # Clean the response to extract only the JSON part
        # Look for a JSON block enclosed in ```json ... ```
        match = re.search(r'```json\s*(\[.*\])\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback to finding the first list
            json_str_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_str_match:
                json_str = json_str_match.group(0)
            else:
                json_str = ""

        if json_str:
            dashboard_def = json.loads(json_str)
            print("Parsed dashboard definition:")
            print(dashboard_def)
            return dashboard_def
        return []
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing dashboard definition from LLM: {e}")
        return []


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
