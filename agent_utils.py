import os
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

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

def create_agent(_df, _llm):
    """Creates a LangChain Pandas DataFrame agent."""
    if _df is None or _llm is None:
        print("🚨 Cannot create agent: DataFrame or LLM is missing.")
        return None
    try:
        agent = create_pandas_dataframe_agent(
            llm=_llm,
            df=_df,
            verbose=True,
            agent_type="tool-calling",
            allow_dangerous_code=True
        )
        print(f"Pandas DataFrame Agent created successfully with tool-calling.")
        return agent
    except Exception as e:
        print(f"🚨 Agent Creation Failed: {e}")
        print("--- Agent Creation Traceback ---")
        print(traceback.format_exc())
        return None
