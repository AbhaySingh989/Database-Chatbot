import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import types

# Define a dummy exception class to act as snowflake.connector.Error
class MockSnowflakeError(Exception):
    pass

# Mock the snowflake.connector library
@pytest.fixture(autouse=True)
def mock_snowflake_connector():
    """
    Mocks the entire 'snowflake' package structure, as the production code
    imports 'snowflake.connector'. This is complex because it's a namespace package.
    """
    # 1. Create the mock for the 'connector' submodule
    mock_connector_module = MagicMock()
    mock_connector_module.Error = MockSnowflakeError

    # 2. Create a fake module for the top-level 'snowflake' package
    snowflake_package = types.ModuleType("snowflake")
    # 3. Set __path__ to make it a package
    snowflake_package.__path__ = ["/mock/path"]
    # 4. Attach the submodule mock to the top-level package mock
    snowflake_package.connector = mock_connector_module

    # 5. Patch sys.modules with BOTH the package and the submodule
    # This ensures that 'import snowflake.connector' works correctly.
    with patch.dict('sys.modules', {
        'snowflake': snowflake_package,
        'snowflake.connector': mock_connector_module,
    }):
        yield mock_connector_module

from database_handler import SnowflakeHandler

@pytest.fixture
def sf_handler():
    return SnowflakeHandler()

def test_connect_success(sf_handler, mock_snowflake_connector):
    """Tests a successful connection to Snowflake."""
    conn_details = {"user": "u", "password": "p", "account": "a"}
    result = sf_handler.connect(**conn_details)
    assert result is True
    assert sf_handler.is_connected is True
    mock_snowflake_connector.connect.assert_called_once_with(**conn_details)

def test_connect_failure(sf_handler, mock_snowflake_connector):
    """Tests a failed connection to Snowflake."""
    mock_snowflake_connector.connect.side_effect = MockSnowflakeError("Connection failed")
    with pytest.raises(ConnectionError, match="Failed to connect to Snowflake"):
        sf_handler.connect()
    assert sf_handler.is_connected is False

def test_disconnect(sf_handler, mock_snowflake_connector):
    """Tests disconnecting from Snowflake."""
    sf_handler.connect()
    sf_handler.disconnect()
    assert sf_handler.is_connected is False
    mock_snowflake_connector.connect.return_value.close.assert_called_once()

def test_get_table_names(sf_handler, mock_snowflake_connector):
    """Tests fetching table names from Snowflake."""
    sf_handler.connect()
    mock_cursor = MagicMock()
    # Snowflake returns a list of tuples, where the table name is the second element
    mock_cursor.fetchall.return_value = [('db', 'table1', 'kind'), ('db', 'table2', 'kind')]
    mock_connection = mock_snowflake_connector.connect.return_value
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    table_names = sf_handler.get_table_names()

    assert table_names == ['table1', 'table2']
    mock_cursor.execute.assert_called_once_with("SHOW TABLES")

@patch('database_handler.pd.read_sql')
def test_execute_query(mock_read_sql, sf_handler, mock_snowflake_connector):
    """Tests executing a query on Snowflake."""
    sf_handler.connect()
    mock_df = pd.DataFrame({'a': [1]})
    mock_read_sql.return_value = mock_df

    query = "SELECT * FROM my_table"
    result = sf_handler.execute_query(query)

    mock_read_sql.assert_called_once_with(query, sf_handler.connection)
    assert result.dataframe is mock_df
