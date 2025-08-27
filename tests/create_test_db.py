import pandas as pd
import sqlite3
import os

def create_test_database():
    """
    Reads data from a CSV and an Excel file and loads them into
    a new SQLite database with two separate tables.
    """
    # Define file paths
    csv_file = 'Testing data/Dream11_DT.csv'
    excel_file = 'Testing data/ipd_simulation_log_v6.xlsx'
    db_file = 'Testing data/test_database.db'

    # Check if source files exist
    if not os.path.exists(csv_file):
        print(f"Error: Source file not found at {csv_file}")
        return
    if not os.path.exists(excel_file):
        print(f"Error: Source file not found at {excel_file}")
        return

    # Remove existing database file to ensure a clean slate
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database file: {db_file}")

    try:
        # Read the source files into pandas DataFrames
        print(f"Reading data from {csv_file}...")
        df_csv = pd.read_csv(csv_file)
        print(f"Successfully read {len(df_csv)} rows from CSV.")

        print(f"Reading data from {excel_file}...")
        df_excel = pd.read_excel(excel_file)
        print(f"Successfully read {len(df_excel)} rows from Excel.")

        # Create a connection to the SQLite database
        print(f"Creating SQLite database at {db_file}...")
        conn = sqlite3.connect(db_file)

        # Write the dataframes to new tables in the database
        print("Writing data to 'dream11' table...")
        df_csv.to_sql('dream11', conn, if_exists='replace', index=False)
        print("...done.")

        print("Writing data to 'ipd_log' table...")
        df_excel.to_sql('ipd_log', conn, if_exists='replace', index=False)
        print("...done.")

        # Close the connection
        conn.close()
        print("Database connection closed.")
        print("\nTest database created successfully with two tables: 'dream11' and 'ipd_log'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_test_database()
