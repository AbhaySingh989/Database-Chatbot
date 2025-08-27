import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from agent_handler import (
    create_database_context_prompt,
    create_enhanced_system_prompt,
    create_agent,
    get_database_suggestions
)

@pytest.fixture
def file_metadata():
    """Fixture for file-based metadata."""
    return {
        'source_type': 'file',
        'file_path': '/path/to/my_data.csv',
        'file_type': 'csv',
        'row_count': 1000,
        'column_count': 10
    }

@pytest.fixture
def db_metadata():
    """Fixture for database-based metadata."""
    return {
        'source_type': 'database',
        'db_path': 'test.db',
        'table_name': 'users',
        'row_count': 5000,
        'column_count': 15,
        'columns': ['id', 'name', 'email', 'age'],
        'column_types': {'id': 'INTEGER', 'name': 'TEXT', 'email': 'TEXT', 'age': 'INTEGER'},
        'foreign_keys': [{'from': 'id', 'table': 'orders', 'to': 'user_id'}],
        'indexes': ['idx_email']
    }

def test_create_db_context_prompt_for_file(file_metadata):
    """Tests that the context prompt for files is generated correctly."""
    prompt = create_database_context_prompt(file_metadata)
    assert "FILE CONTEXT" in prompt
    assert "Source File: my_data.csv" in prompt
    assert "File Type: CSV" in prompt
    assert "Total Rows: 1,000" in prompt

def test_create_db_context_prompt_for_db(db_metadata):
    """Tests that the context prompt for databases is generated correctly."""
    prompt = create_database_context_prompt(db_metadata)
    assert "DATABASE CONTEXT" in prompt
    assert "Database: test.db" in prompt
    assert "Source Table: users" in prompt
    assert "SCHEMA INFORMATION" in prompt
    assert "- name: TEXT" in prompt
    assert "RELATIONSHIPS" in prompt
    assert "- id → orders.user_id" in prompt
    assert "INDEXES" in prompt
    assert "idx_email" in prompt

def test_create_enhanced_system_prompt(db_metadata):
    """Tests the full system prompt creation."""
    prompt = create_enhanced_system_prompt(db_metadata)
    # Check for base instructions
    assert "You are a helpful data analysis assistant." in prompt
    assert "IMPORTANT NOTE ON PYTHON_REPL TOOL" in prompt
    # Check that the DB context is included
    assert "DATABASE CONTEXT" in prompt
    assert "DATABASE-SPECIFIC GUIDANCE" in prompt

@patch('agent_handler.create_pandas_dataframe_agent')
def test_create_agent_success(mock_create_pandas_agent, db_metadata):
    """Tests the successful creation of an agent."""
    mock_llm = MagicMock()
    mock_df = pd.DataFrame({'a': [1]})

    # The mock should return a mock agent
    mock_agent = MagicMock()
    mock_create_pandas_agent.return_value = mock_agent

    agent = create_agent(mock_df, mock_llm, metadata=db_metadata)

    assert agent is mock_agent
    # Verify that the underlying langchain function was called with the correct arguments
    mock_create_pandas_agent.assert_called_once()
    call_args = mock_create_pandas_agent.call_args
    assert call_args.kwargs['llm'] is mock_llm
    assert call_args.kwargs['df'] is mock_df
    assert "DATABASE CONTEXT" in call_args.kwargs['prefix']

def test_create_agent_with_none_df():
    """Tests that agent creation fails if the dataframe is None."""
    agent = create_agent(None, MagicMock(), metadata={})
    assert agent is None

def test_get_db_suggestions_for_db(db_metadata):
    """Tests that database-specific suggestions are generated."""
    suggestions = get_database_suggestions(db_metadata)
    assert len(suggestions) > 0
    # Check for a suggestion related to the table name 'users'
    assert any("user/customer demographics" in s for s in suggestions)

def test_get_db_suggestions_for_file(file_metadata):
    """Tests that no suggestions are generated for file-based data."""
    suggestions = get_database_suggestions(file_metadata)
    assert isinstance(suggestions, list)
    assert len(suggestions) == 0

def test_get_proactive_suggestions(db_metadata):
    """Tests the LLM-based proactive suggestion generation."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """
    1. What is the distribution of user ages?
    2. How many orders has each user placed?
    3. Is there a correlation between age and number of orders?
    """
    mock_llm.invoke.return_value = mock_response

    from agent_handler import get_proactive_suggestions
    suggestions = get_proactive_suggestions(db_metadata, mock_llm)

    assert len(suggestions) == 3
    assert suggestions[0] == "What is the distribution of user ages?"
    assert suggestions[2] == "Is there a correlation between age and number of orders?"
    mock_llm.invoke.assert_called_once()

def test_generate_dashboard_definition_success(db_metadata):
    """Tests that a valid dashboard JSON with layout is parsed correctly."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    json_payload = """
    [
      {"type": "metric", "label": "Avg. Age", "column": "age", "expression": "AVERAGE", "x": 0, "y": 0, "w": 4, "h": 1},
      {"type": "bar_chart", "label": "Orders by City", "x": "city", "y": "orders", "x": 4, "y": 0, "w": 8, "h": 2}
    ]
    """
    mock_response.content = f"```json\n{json_payload}\n```"
    mock_llm.invoke.return_value = mock_response

    from agent_handler import generate_dashboard_definition
    dashboard_def = generate_dashboard_definition(db_metadata, mock_llm)

    assert len(dashboard_def) == 2
    assert dashboard_def[0]['type'] == 'metric'
    assert 'w' in dashboard_def[0]
    assert dashboard_def[1]['label'] == 'Orders by City'
    assert dashboard_def[1]['h'] == 2

def test_generate_dashboard_definition_failure(db_metadata):
    """Tests that invalid JSON from the LLM is handled gracefully."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "I am sorry, I cannot create a JSON."
    mock_llm.invoke.return_value = mock_response

    from agent_handler import generate_dashboard_definition
    dashboard_def = generate_dashboard_definition(db_metadata, mock_llm)

    assert isinstance(dashboard_def, list)
    assert len(dashboard_def) == 0
