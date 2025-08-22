import streamlit as st
import pandas as pd

@st.cache_data(show_spinner="Loading files...")
def load_and_combine_files(uploaded_files):
    """
    Loads data from uploaded files (CSV, Excel).
    Returns a list of dictionaries, each containing 'name' and 'df', or None if no files.
    """
    if not uploaded_files:
        return None

    loaded_dfs_info = []
    file_details_summary = [] # For the summary table

    for file in uploaded_files:
        file_name = file.name

        try:
            df = None
            file.seek(0) # Reset file pointer
            if file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    file_details_summary.append({"name": file_name, "type": "CSV", "rows": len(df)})
                except pd.errors.ParserError as e:
                    st.error(f"Error parsing CSV file '{file_name}': {e}")
                    file_details_summary.append({"name": file_name, "type": "CSV", "status": f"ParserError: {e}"})
            elif file_name.endswith(('.xls', '.xlsx')):
                try:
                    df = pd.read_excel(file, engine='openpyxl')
                    file_details_summary.append({"name": file_name, "type": "Excel", "rows": len(df)})
                except Exception as e:
                    st.error(f"Error reading Excel file '{file_name}': {e}")
                    file_details_summary.append({"name": file_name, "type": "Excel", "status": f"Error: {e}"})
            else:
                 st.warning(f"Unsupported file type: {file_name}. Skipping.")
                 file_details_summary.append({"name": file_name, "type": "Unsupported", "status": "Skipped"})

            if df is not None:
                loaded_dfs_info.append({'name': file_name, 'df': df})

        except FileNotFoundError as e:
            st.error(f"Error: File not found '{file_name}': {e}")
            file_details_summary.append({"name": file_name, "status": f"FileNotFoundError: {e}"})
        except Exception as e:
            st.error(f"An unexpected error occurred while processing '{file_name}': {e}")
            file_details_summary.append({"name": file_name, "status": f"Unexpected Error: {e}"})

    st.subheader("File Loading Summary")
    st.dataframe(pd.DataFrame(file_details_summary))

    if not loaded_dfs_info:
        st.warning("No valid dataframes were loaded.")
        return None

    return loaded_dfs_info
