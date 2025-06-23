from agentD import tools
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from openai import OpenAIError
from rdkit import Chem
import requests
import os
import yaml
import ast
import importlib.util
import pandas as pd

# from dZiner paper
def RetrievalQABypassTokenLimit(vector_store, RetrievalQA_prompt, llm, k=10, fetch_k=50, min_k=2, chain_type="stuff"):
    while k >= min_k:
        try:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k},
            )
            qa_chain = RetrievalQA.from_chain_type( 
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                memory=ConversationBufferMemory(
                        ))
            
            # Check to see if we hit the token limit
            result = qa_chain.run({"query": RetrievalQA_prompt})
            return result  # If successful, return the result and exit the function

        except OpenAIError as e:
            # Check if it's a token limit error
            print(e)
            if 'maximum context length' in str(e):
                print(f"\nk={k} results hitting the token limit. Reducing k and retrying...\n")
                k -= 1
            else:
                # Re-raise other OpenAI errors
                raise e
            

def download_pdf(url: str, title: str, paper_dir: str):
    """
    Downloads a PDF from a given URL and saves it with a sanitized title.

    **Args:**
        url (str): The URL of the PDF.
        title (str): The title of the paper.

    **Returns:**
        str: The file path of the downloaded PDF.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Sanitize filename
            filename = "".join(c if c.isalnum() else "_" for c in title)[:100] + ".pdf"
            # create PAPER_DIR if it doesn't exist
            if not os.path.exists(paper_dir):
                os.makedirs(paper_dir)
            file_path = os.path.join(paper_dir, filename)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            # print(f"Downloaded: {file_path}")
            return file_path
        else:
            print(f"Failed to download: {title} ({url})")
            return None#f"Failed to download: {title} ({url})"
        #     pass
    except Exception as e:
        print(f"Error downloading {title}: {str(e)}")
        return None #f"Error downloading {title}: {str(e)}"
    

def get_tool_decorated_functions(filepath):
    """
    Parse a Python file, find all functions decorated with @tool,
    dynamically import the module, and return the function objects.

    Args:
        filepath (str): Path to the Python file to analyze.

    Returns:
        list: List of function objects decorated with @tool.
    """
    # Step 1: Parse the AST and find decorated functions
    with open(filepath, "r", encoding="utf-8") as f:
        node = ast.parse(f.read(), filename=filepath)

    decorated_func_names = []
    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                if isinstance(d, ast.Name) and d.id == "tool":
                    decorated_func_names.append(n.name)
                elif isinstance(d, ast.Call) and getattr(d.func, "id", "") == "tool":
                    decorated_func_names.append(n.name)

    # Step 2: Import the module dynamically
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Step 3: Extract function objects
    tools = [getattr(module, name) for name in decorated_func_names]

    return tools


def custom_serializer(obj):
    """
    Custom serializer for objects that are not JSON serializable.
    Converts the object to string, or returns a placeholder for non-serializable objects.

    Args:
        obj: Any Python object.

    Returns:
        str: String representation of the object or a placeholder.
    """
    try:
        return str(obj)
    except:
        return f"<<Non-serializable: {type(obj).__name__}>>"
    

def process_dataset_with_agent(file_path: str, protein: str, tools: list, agent) -> pd.DataFrame:
    """
    Reads the dataset and applies the LLM-driven analysis to each entry.

    Args:
        file_path (str): Path to the dataset file.
        protein (str): Target protein for drug analysis.
        objective (str): Target objective for drug analysis.
        tools (list): List of tool objects to be used by the agent.
        agent: The LLM agent object.

    Returns:
        pd.DataFrame: DataFrame with the original data and LLM assessments.
    """
    # Read dataset
    df = pd.read_csv(file_path)

    # Prepare list to store output
    results = []

    # Iterate through each row
    for _, row in df.iterrows():

        entry = row.to_dict()

        keys_to_remove = [key for key in entry.keys() if "Probability" in key or "Interpretation" in key]
        for key in keys_to_remove:
            entry.pop(key, None)

        smiles = row['SMILES']
        prompt = (
            f"Propose a modified version of the given SMILES that addresses its weakness as a drug candidate. "
            f"Consider that the given molecule is targeting {protein}. "
            f"The molecule's properties are: {entry}."
        )

        tool_names = [tool.name for tool in tools]  
        tool_desc = [tool.description for tool in tools]
        input_data = {
            "input": prompt,
            "tools": tools,
            "tool_names": tool_names,
            "tool_desc": tool_desc
        }
        
        try:
            response = agent.invoke(input_data)
            assessment = response.get('output', 'No response')

            updated_smiles, property, rationale = assessment.split('</s>')
            results.append({'SMILES': smiles, 'Updated_SMILES': updated_smiles,'Property': property, 'Rationale': rationale})

        except Exception as e:
            assessment = f"Error: {str(e)}"

    # Create and return the new DataFrame
    return pd.DataFrame(results)

