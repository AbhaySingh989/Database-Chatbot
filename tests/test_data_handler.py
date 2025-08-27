import pytest
import pandas as pd
from unittest.mock import patch, call
from data_handler import load_and_combine_files # Import the decorated function
import io

# This fixture will mock the 'st' object used within the data_handler module.
@pytest.fixture
def mock_st():
    """Mocks the 'st' object in data_handler to isolate the logic."""
    with patch('data_handler.st') as mock_streamlit:
        yield mock_streamlit

def create_mock_file(content, name):
    """Creates a mock file object that simulates Streamlit's UploadedFile."""
    if isinstance(content, str):
        content = content.encode('utf-8')

    mock_file = io.BytesIO(content)
    mock_file.name = name
    mock_file.seek(0)
    return mock_file

@pytest.fixture
def valid_csv_file():
    """Fixture for a valid CSV file mock."""
    with open('Testing data/Dream11_DT.csv', 'rb') as f:
        content = f.read()
    return create_mock_file(content, 'Dream11_DT.csv')

@pytest.fixture
def valid_excel_file():
    """Fixture for a valid Excel file mock."""
    with open('Testing data/ipd_simulation_log_v6.xlsx', 'rb') as f:
        content = f.read()
    return create_mock_file(content, 'ipd_simulation_log_v6.xlsx')

# We test the __wrapped__ function to bypass the streamlit decorator for unit testing.
def test_load_single_csv_file(valid_csv_file, mock_st):
    """Tests loading a single, valid CSV file."""
    uploaded_files = [valid_csv_file]
    # Call the original, undecorated function
    result = load_and_combine_files.__wrapped__(uploaded_files)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1

    info = result[0]
    assert info['name'] == 'Dream11_DT.csv'
    assert isinstance(info['df'], pd.DataFrame)
    assert not info['df'].empty
    # CORRECTED: Assert for a column that actually exists.
    assert 'home_team' in info['df'].columns
    # Check that streamlit UI functions were called
    mock_st.subheader.assert_called_with("File Loading Summary")
    mock_st.dataframe.assert_called()


def test_load_single_excel_file(valid_excel_file, mock_st):
    """Tests loading a single, valid Excel file."""
    uploaded_files = [valid_excel_file]
    result = load_and_combine_files.__wrapped__(uploaded_files)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1

    info = result[0]
    assert info['name'] == 'ipd_simulation_log_v6.xlsx'
    assert isinstance(info['df'], pd.DataFrame)
    assert not info['df'].empty

def test_load_multiple_files(valid_csv_file, valid_excel_file, mock_st):
    """Tests loading both a CSV and an Excel file together."""
    uploaded_files = [valid_csv_file, valid_excel_file]
    result = load_and_combine_files.__wrapped__(uploaded_files)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]['name'] == 'Dream11_DT.csv'
    assert result[1]['name'] == 'ipd_simulation_log_v6.xlsx'

def test_load_no_files(mock_st):
    """Tests calling the function with an empty list."""
    # The decorator would normally handle this, but we test the raw function
    result = load_and_combine_files.__wrapped__([])
    assert result is None
    # No UI should be displayed for no files
    mock_st.subheader.assert_not_called()

def test_load_unsupported_file_type(mock_st):
    """Tests loading a file with an unsupported extension."""
    unsupported_file = create_mock_file("some text data", "test.txt")
    result = load_and_combine_files.__wrapped__([unsupported_file])
    assert result is None
    # CORRECTED: Check that the specific warning was called, among others.
    expected_warning = call('Unsupported file type: test.txt. Skipping.')
    assert expected_warning in mock_st.warning.call_args_list

# CORRECTED: Test the exception block directly by mocking the reader.
@patch('data_handler.pd.read_csv', side_effect=pd.errors.ParserError("Test parsing error"))
def test_load_malformed_csv(mock_read_csv, mock_st):
    """Tests that a ParserError is handled correctly."""
    bad_file = create_mock_file("this content will be ignored by the mock", "bad.csv")
    result = load_and_combine_files.__wrapped__([bad_file])

    assert result is None
    mock_st.error.assert_called()
    # Check that the error message contains the expected text
    error_message = mock_st.error.call_args[0][0]
    assert "Error parsing CSV file 'bad.csv'" in error_message
