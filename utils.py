import os
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from itertools import combinations

def get_llm(model_name, temperature):
    """Creates and returns a ChatGoogleGenerativeAI instance."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        print(f"LLM '{model_name}' initialized.")
        return llm
    except Exception as e:
        print(f"🚨 Failed to initialize LLM '{model_name}': {e}")
        print(f"--- LLM Init Traceback ({model_name}) ---")
        print(traceback.format_exc())
        raise e

def find_potential_common_columns(list_of_df_infos):
    """
    Finds columns with exact matching names between all unique pairs of DataFrames.
    Input: list_of_df_infos (e.g., st.session_state.individual_dfs)
    Output: Dictionary {('df_name1', 'df_name2'): ['common_col1', ...], ...}
    """
    if not list_of_df_infos or len(list_of_df_infos) < 2:
        return {}

    common_columns_candidates = {}

    for (info1, info2) in combinations(list_of_df_infos, 2):
        df1_name = info1['name']
        df1 = info1['df']
        df2_name = info2['name']
        df2 = info2['df']

        common_cols = list(set(df1.columns) & set(df2.columns))
        if common_cols:
            common_columns_candidates[(df1_name, df2_name)] = common_cols
    return common_columns_candidates
