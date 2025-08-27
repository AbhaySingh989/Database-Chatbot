import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Define a dummy exception class to act as psycopg2.Error for our tests
class MockPsycopg2Error(Exception):
    pass

# Mock the psycopg2 library before it's imported by the handler
@pytest.fixture(autouse=True)
def mock_psycopg2():
    """Mocks the psycopg2 library and its Error class."""
    mock_psycopg2_module = MagicMock()
    mock_psycopg2_module.Error = MockPsycopg2Error
    with patch.dict('sys.modules', {'psycopg2': mock_psycopg2_module}):
        yield mock_psycopg2_module

from database_handler import PostgreSQLHandler, ColumnInfo, TableInfo

@pytest.fixture
def pg_handler():
    """Fixture to provide a PostgreSQLHandler instance."""
    return PostgreSQLHandler()

def test_connect_success(pg_handler, mock_psycopg2):
    """Tests a successful connection to PostgreSQL."""
    conn_details = {"dbname": "test", "user": "user", "password": "pw", "host": "host", "port": 5432}

    # The connect call should succeed
    result = pg_handler.connect(**conn_details)

    assert result is True
    assert pg_handler.is_connected is True
    mock_psycopg2.connect.assert_called_once_with(**conn_details)

def test_connect_failure(pg_handler, mock_psycopg2):
    """Tests a failed connection to PostgreSQL."""
    conn_details = {"dbname": "test", "user": "user", "password": "pw", "host": "host", "port": 5432}

    # Configure the mock to raise our custom exception
    mock_psycopg2.connect.side_effect = MockPsycopg2Error("Connection failed")

    with pytest.raises(ConnectionError, match="Failed to connect to PostgreSQL: Connection failed"):
        pg_handler.connect(**conn_details)

    assert pg_handler.is_connected is False

def test_disconnect(pg_handler, mock_psycopg2):
    """Tests disconnecting from PostgreSQL."""
    # First, simulate a successful connection
    pg_handler.connect()
    assert pg_handler.is_connected is True

    # Now, disconnect
    pg_handler.disconnect()

    assert pg_handler.is_connected is False
    assert pg_handler.connection is None
    # Ensure the mock connection's close method was called
    mock_psycopg2.connect.return_value.close.assert_called_once()

def test_get_table_names(pg_handler, mock_psycopg2):
    """Tests fetching table names."""
    # Simulate a connection
    pg_handler.connect()

    # Configure the mock cursor and its return value
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [('users',), ('products',)]
    mock_connection = mock_psycopg2.connect.return_value
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    table_names = pg_handler.get_table_names()

    assert table_names == ['users', 'products']
    mock_cursor.execute.assert_called_once()

@patch('database_handler.pd.read_sql_query')
def test_execute_query(mock_read_sql, pg_handler, mock_psycopg2):
    """Tests executing a query."""
    pg_handler.connect()

    mock_df = pd.DataFrame({'a': [1, 2]})
    mock_read_sql.return_value = mock_df

    query = "SELECT * FROM users"
    result = pg_handler.execute_query(query)

    mock_read_sql.assert_called_once_with(query, pg_handler.connection, params=None)
    assert result.row_count == 2
    assert result.dataframe is mock_df
