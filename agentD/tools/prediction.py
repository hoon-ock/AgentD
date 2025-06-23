from langchain.tools import tool
import torch
from rdkit import Chem
import re, json, requests, time, os, tqdm, random
import pandas as pd
from agentD.tools.BAPULM import BAPULM, EmbeddingExtractor, set_seed
from configs.tool_globals import ADMET_WAIT_INTERVAL, ADMET_MAX_WAIT_TIME, PROPERTY_DIR

random.seed(2102)

# Adapted with modifications from the original implementation at:
# https://github.com/mehradans92/dZiner/blob/main/dziner/tools.py
@tool
def check_smiles_validity(smiles: str):
    '''
    Checks if the provided SMILES string is valid.
    '''
    try:
        smiles = smiles.strip()
        smiles = smiles.replace("\n", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES" 
        return "Valid SMILES" 
    except Exception as e:
        return f"Error checking SMILES validity: {str(e)}"


@tool
def predict_affinity_batch(data: str):
    """
    Predicts binding affinities for multiple entries given a list of SMILES strings and their indices.
    To run this tool, you need to provide a JSON dictionary with "sequence" and "smiles_path" keys.
    "smiles_path" should be a path to a CSV file containing a column named "SMILES".
    Args:
        data: valid JSON dictionary with "sequence" (str) and "smiles_path" (str).

    """
    # Validate input
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Please provide a valid JSON string."
    if "sequence" not in data:
        return "Error: Missing required input. Provide 'sequence'."
    if "smiles_path" not in data:
        return "Error: Missing required input. Provide 'smiles_path'."

    sequence = data.get("sequence")
    smiles_file_path = data.get("smiles_path")

    # Load the SMILES data
    try:
        df_smiles = pd.read_csv(smiles_file_path)
        smiles_list = df_smiles["SMILES"].tolist()
    except Exception as e:
        return f"Error loading SMILES file: {str(e)}"
    
    # checkpoint path
    model_checkpoint = os.path.join(os.path.dirname(__file__),
                                    "BAPULM_results_molformer_reproduce_json.pth")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model = BAPULM().to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Initialize Embedding Extractor
    extractor = EmbeddingExtractor(device)

    # Output list for the results
    results = []

    # Preprocess the sequence
    sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))

    # Iterate through the dataset
    #for idx, smiles in zip(index_list, smiles_list):
    for smiles in tqdm.tqdm(smiles_list):
        try:
            # Extract embeddings
            with torch.no_grad():
                prot_embedding, mol_embedding = extractor.get_combined_embedding(sequence, smiles)

            # Move embeddings to device
            prot_embedding = prot_embedding.to(device)
            mol_embedding = mol_embedding.to(device)

            # Run inference
            with torch.no_grad():
                affinity = model(prot_embedding, mol_embedding)

            # Rescale the output affinity (from the reference model)
            mean = 6.51286529169358 
            scale = 1.5614094578916633
            affinity = (affinity * scale) + mean

            # Collect results
            # results.append({"Index": idx, "SMILES": smiles, "Affinity": affinity.item()})
            results.append({"SMILES": smiles, "Affinity [pKd]": affinity.item()})
        except Exception as e:
            # Handle any exception and log as an error entry
            results.append({"SMILES": smiles, "Affinity [pKd]": f"Error: {str(e)}"})
    # Convert results to DataFrame and save as CSV
    os.makedirs(PROPERTY_DIR, exist_ok=True)
    output_path = os.path.join(PROPERTY_DIR, "affinity_"+smiles_file_path.split("/")[-1])
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    return "Affinity predictions saved to ", output_path


@tool
def get_admet_predictions(csv_file_path):
    """
    This tool predicts ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) 
    properties by interacting with the DeepPK web API. 

    Arguments:
    - csv_file_path (str): Path to a CSV file. e.g. "pool/combined_smiles.csv"

   """
    
    submit_url = "https://biosig.lab.uq.edu.au/deeppk/api/predict"

    # Step 1: Submit the file
    try:
        with open(csv_file_path, 'rb') as f:
            files = {'smiles_file': f}
            data = {'pred_type': 'admet'}
            response = requests.post(submit_url, files=files, data=data)
            response.raise_for_status()
            job_id = json.loads(response.text).get("job_id")
            if not job_id:
                return {"error": "Job ID not returned from submission."}
    except Exception as e:
        return {"error": f"Submission failed: {e}"}

    # Step 2: Poll using multipart/form-data GET like curl -F
    elapsed = 0
    session = requests.Session()
    while elapsed < ADMET_MAX_WAIT_TIME:
        try:
            multipart_form_data = {'job_id': (None, job_id)}
            request = requests.Request('GET', submit_url, files=multipart_form_data)
            prepared = session.prepare_request(request)
            response = session.send(prepared)
            if response.status_code == 500:
                return {"error": f"500 Server Error. Job ID: {job_id}"}
            response.raise_for_status()
            result_json = json.loads(response.text)    
            # If it’s the running status
            if isinstance(result_json, dict) and result_json.get("status") == "running":
                # print("Still running...")
                time.sleep(ADMET_WAIT_INTERVAL)
                elapsed += ADMET_WAIT_INTERVAL
                continue

            # If it's a string, decode again
            if isinstance(result_json, str):
                result_json = json.loads(result_json)
                df = pd.DataFrame(result_json).T
                os.makedirs(PROPERTY_DIR, exist_ok=True)
                output_path = os.path.join(PROPERTY_DIR, "admet_"+csv_file_path.split("/")[-1])
                df.to_csv(output_path, index=False)

            return f"ADMET predictions saved to {output_path}"
        except Exception as e:
            return {"error": f"Polling failed: {e}"}