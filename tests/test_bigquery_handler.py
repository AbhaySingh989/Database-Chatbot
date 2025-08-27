import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Mock the google.cloud.bigquery library
@pytest.fixture(autouse=True)
def mock_bigquery():
    mock_bigquery_module = MagicMock()
    mock_credentials = MagicMock()
    mock_credentials.project_id = "test-project"

    with patch.dict('sys.modules', {
        'google.cloud.bigquery': mock_bigquery_module,
        'google.oauth2.service_account': MagicMock()
    }):
        # Mock the from_service_account_file method to return mock credentials
        from google.oauth2 import service_account
        service_account.Credentials.from_service_account_file.return_value = mock_credentials
        yield mock_bigquery_module

from database_handler import BigQueryHandler

@pytest.fixture
def bq_handler():
    return BigQueryHandler()

def test_connect_success(bq_handler, mock_bigquery):
    """Tests a successful connection to BigQuery."""
    result = bq_handler.connect(service_account_json_path="dummy/path.json")
    assert result is True
    assert bq_handler.is_connected is True
    mock_bigquery.Client.assert_called_once()

def test_connect_failure(bq_handler, mock_bigquery):
    """Tests a failed connection to BigQuery."""
    from google.oauth2 import service_account
    service_account.Credentials.from_service_account_file.side_effect = Exception("File not found")

    with pytest.raises(ConnectionError, match="Failed to connect to BigQuery"):
        bq_handler.connect(service_account_json_path="dummy/path.json")

    assert bq_handler.is_connected is False

def test_get_table_names(bq_handler, mock_bigquery):
    """Tests fetching table names from BigQuery."""
    bq_handler.connect(service_account_json_path="dummy/path.json")

    mock_table1 = MagicMock()
    mock_table1.table_id = "table1"
    mock_table2 = MagicMock()
    mock_table2.table_id = "table2"

    mock_client = mock_bigquery.Client.return_value
    mock_client.list_tables.return_value = [mock_table1, mock_table2]

    table_names = bq_handler.get_table_names()

    assert table_names == ["table1", "table2"]
    mock_client.list_tables.assert_called_once()

def test_execute_query(bq_handler, mock_bigquery):
    """Tests executing a query on BigQuery."""
    bq_handler.connect(service_account_json_path="dummy/path.json")

    mock_df = pd.DataFrame({'col1': [1, 2, 3]})
    mock_query_job = MagicMock()
    mock_query_job.to_dataframe.return_value = mock_df

    mock_client = mock_bigquery.Client.return_value
    mock_client.query.return_value = mock_query_job

    query = "SELECT * FROM my_table"
    result = bq_handler.execute_query(query)

    assert result.dataframe is mock_df
    mock_client.query.assert_called_once_with(query)
    mock_query_job.to_dataframe.assert_called_once()
