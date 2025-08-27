import pytest
import os
import pandas as pd
from database_handler import SQLiteHandler, ColumnInfo, TableInfo, QueryResult

DB_PATH = 'Testing data/test_database.db'

@pytest.fixture
def db_handler():
    """Fixture to provide a connected SQLiteHandler instance and ensure cleanup."""
    # Pre-condition check: Ensure the test database exists.
    if not os.path.exists(DB_PATH):
        pytest.fail(f"Test database '{DB_PATH}' not found. Please run create_test_db.py first.")

    handler = SQLiteHandler()
    handler.connect(DB_PATH)
    yield handler
    # Teardown: ensure connection is closed after each test.
    handler.disconnect()

def test_connection_success(db_handler):
    """Tests successful connection to the database."""
    assert db_handler.is_connected is True
    assert db_handler.connection is not None
    assert db_handler.db_path == DB_PATH

def test_connection_to_non_existent_db():
    """Tests that connecting to a non-existent file raises an error."""
    handler = SQLiteHandler()
    with pytest.raises(ConnectionError):
        handler.connect('non_existent_database.db')
    assert handler.is_connected is False

def test_disconnect(db_handler):
    """Tests the disconnect method."""
    assert db_handler.is_connected is True
    db_handler.disconnect()
    assert db_handler.is_connected is False
    assert db_handler.connection is None
    assert db_handler.db_path is None

def test_get_table_names(db_handler):
    """Tests fetching all table names from the database."""
    table_names = db_handler.get_table_names()
    assert isinstance(table_names, list)
    # The order might not be guaranteed, so use a set for comparison.
    assert set(table_names) == {'dream11', 'ipd_log'}

def test_get_table_row_count(db_handler):
    """Tests fetching the row count for a specific table."""
    # From create_test_db.py, we know ipd_log has 641 rows.
    count = db_handler.get_table_row_count('ipd_log')
    assert count == 641

def test_get_table_schema(db_handler):
    """Tests fetching the schema for a specific table."""
    schema = db_handler.get_table_schema('dream11')
    assert isinstance(schema, list)
    assert len(schema) > 0
    assert all(isinstance(col, ColumnInfo) for col in schema)

    # Check for a known column
    season_column = next((col for col in schema if col.name == 'season'), None)
    assert season_column is not None
    assert season_column.data_type == 'INTEGER'

def test_get_table_info(db_handler):
    """Tests getting comprehensive info for a table."""
    info = db_handler.get_table_info('dream11')
    assert isinstance(info, TableInfo)
    assert info.name == 'dream11'
    assert info.row_count == 22362
    assert len(info.columns) > 0
    assert info.columns[0].name == 'season'

def test_execute_valid_query(db_handler):
    """Tests executing a simple, valid SELECT query."""
    query = "SELECT fullName, Batting_FP FROM dream11 WHERE Batting_FP > 100 LIMIT 5"
    result = db_handler.execute_query(query)

    assert isinstance(result, QueryResult)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert result.row_count == 5
    assert len(result.dataframe) == 5
    assert 'fullName' in result.columns
    assert result.query == query

def test_execute_query_security_violation(db_handler):
    """Tests that a query with a forbidden keyword raises a ValueError."""
    # The decorator should catch this before execution.
    unsafe_query = "DROP TABLE dream11"
    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        db_handler.execute_query(unsafe_query)

def test_execute_query_invalid_syntax(db_handler):
    """Tests that a query with invalid SQL syntax raises an error."""
    invalid_query = "SELECT FROM dream11 WHERE"
    # The internal validation should catch this.
    with pytest.raises(RuntimeError, match="Query execution failed: Query validation failed: SQL syntax error"):
        db_handler.execute_query(invalid_query)

def test_get_query_history(db_handler):
    """Tests the query history functionality."""
    # Clear history for a clean test
    db_handler.query_history = []

    q1 = "SELECT * FROM ipd_log LIMIT 1"
    q2 = "SELECT * FROM dream11 LIMIT 1"

    db_handler.execute_query(q1)
    db_handler.execute_query(q2)

    history = db_handler.get_query_history()
    assert len(history) == 2
    assert history[0]['query'] == q1
    assert history[0]['success'] is True
    assert history[1]['query'] == q2
    assert history[1]['success'] is True
