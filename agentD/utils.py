from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from openai import OpenAIError
from rdkit import Chem
import requests
import os
import yaml

# BOLTZ_YAML_PATH = "configs/boltz.yaml"

# current directory
# DIR_NAME = 'papers'
# script_dir = os.path.dirname(__file__)
# PAPER_DIR = "/home/hoon/dd-agent/llm_dd/agentD/papers" #os.path.join(script_dir, DIR_NAME)


# def drug_chemical_feasibility(smiles: str):
    # '''
    # Evaluates the chemical feasibility of a drug candidate based on its SMILES string.
    # This function checks whether the provided SMILES string is valid and can be converted into a molecular structure.
    # '''
    # smiles = smiles.replace("\n", "")
    # mol = Chem.MolFromSmiles(smiles)
    # if mol is None:
    #     return "Invalid SMILES"
    # return "Valid SMILES"

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
    


