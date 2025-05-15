from langchain.tools import tool
import torch
from rdkit import Chem
from rdkit.Chem import QED
import re, json, requests, time, sys, os, tqdm, random
import pandas as pd

cwd = os.getcwd()
home_dir = os.path.dirname(os.path.dirname(cwd))
sys.path.append(home_dir)
from agentD.tools.BAPULM import BAPULM, EmbeddingExtractor, set_seed
from agentD.tools import sascorer

import warnings
warnings.filterwarnings('ignore')

CHECKPOINT_PATH = "agentD/tools/BAPULM_results_molformer_reproduce_json.pth"
ADMET_WAIT_INTERVAL=30
ADMET_MAX_WAIT_TIME=300
# set random seed
random.seed(2102)


# from https://github.com/mehradans92/dZiner/blob/main/dziner/tools.py
@tool
def drug_chemical_feasibility(smiles: str):
    '''
    This tool inputs a SMILES of a drug candidate and outputs chemical feasibility, Synthetic Accessibility (SA),
      and Quantitative drug-likeness(QED) scores. SA the ease of synthesis of compounds
        according to their synthetic complexity which combines starting materials information and structural complexity. 
        Lower SA, means better synthesiability. QED combines eight physicochemical properties(molecular weight, LogP, 
        H-bond donors, H-bond acceptors, charge, aromaticity, stereochemistry and solubility), generating a score between 0 and 1.
    '''
    smiles = smiles.replace("\n", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES", 0 ,0

    # Calculate SA score
    sa_score = sascorer.calculateScore(mol)

    # Optionally calculate QED score
    # molecular_weight = Descriptors.MolWt(mol)
    qed_score = QED.qed(mol)
    return "Valid SMILES", sa_score, qed_score

# @tool
# def predict_affinity(data: str):
#     """
#     Predicts the binding affinity given a protein sequence and ligand SMILES string.
#     To run this tool, you need to provide a dictionary with "sequence" and "smiles" keys.
#     The sequence should be a string representing the protein sequence, and smiles should be a string
#     representing the ligand SMILES string. The function will return the predicted binding affinity.

#     Args:
#         data: valid JSON Dictionary with "sequence" (str) and "smiles" (str).

#     Returns:
#         float: Predicted binding affinity.
#     """
#     set_seed(2102) # seed used in the original BAPULM paper

#     # Convert JSON string to dictionary if needed
#     if isinstance(data, str):
#         try:
#             data = json.loads(data)
#         except json.JSONDecodeError:
#             return "Error: Invalid input format. Expected JSON with 'sequence' and 'smiles'."

#     sequence = data.get("sequence")
#     smiles = data.get("smiles")

#     if not sequence or not smiles:
#         return "Error: Missing required inputs. Provide both 'sequence' and 'smiles'."

#     # checkpoint path
#     model_checkpoint = os.path.join(home_dir, CHECKPOINT_PATH)#"agentD/tools/BAPULM_results_molformer_reproduce_json.pth")
#     # "agentD/tools/BAPULM_results_molformer_reproduce_json.pth"
#     #"/home/hoon/dd-agent/dd-agent/tools/BAPULM_results_molformer_reproduce_json.pth"
#     # Ensure correct device
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # print(device)
#     # Load Model
#     model = BAPULM().to(device)
#     checkpoint = torch.load(model_checkpoint, map_location=device)
#     model.load_state_dict(checkpoint)
#     model.eval()

#     # Initialize Embedding Extractor
#     extractor = EmbeddingExtractor(device)

#     # Preprocess the sequence (replace uncommon amino acids)
#     sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))

#     # Extract embeddings
#     with torch.no_grad():
#         prot_embedding, mol_embedding = extractor.get_combined_embedding(sequence, smiles)

#     # Move embeddings to device
#     prot_embedding = prot_embedding.to(device)
#     mol_embedding = mol_embedding.to(device)

#     # Run inference
#     with torch.no_grad():
#         affinity = model(prot_embedding, mol_embedding)

#     # Resacle the output affinity
#     mean = 6.51286529169358
#     scale = 1.5614094578916633
#     affinity = (affinity * scale) + mean

#     # Return binding affinity score
#     return affinity.item()


@tool
def predict_affinity_batch(data: str):
    """
    Predicts binding affinities for multiple entries given a list of SMILES strings and their indices.
    To run this tool, you need to provide a JSON dictionary with "sequence" and "smiles_path" keys.
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
        # index_list = df_smiles["Index"].tolist()
    except Exception as e:
        return f"Error loading SMILES file: {str(e)}"
    
    # checkpoint path
    model_checkpoint = os.path.join(home_dir, CHECKPOINT_PATH)
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

            # Rescale the output affinity
            mean = 6.51286529169358
            scale = 1.5614094578916633
            affinity = (affinity * scale) + mean

            # Collect results
            # results.append({"Index": idx, "SMILES": smiles, "Affinity": affinity.item()})
            results.append({"SMILES": smiles, "Affinity [pKa]": affinity.item()})
        except Exception as e:
            # Handle any exception and log as an error entry
            #results.append({"Index": idx, "SMILES": smiles, "Affinity": f"Error: {str(e)}"})
            results.append({"SMILES": smiles, "Affinity [pKa]": f"Error: {str(e)}"})
    # Convert results to DataFrame and save as CSV
    # output_path = os.path.join("property", "affinity_"+smiles_file_path.split("/")[-1])
    output_path = os.path.join("property", "affinity.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    # print("Affinity predictions saved to ", output_path)
    return "Affinity predictions saved to ", output_path
    # return df

@tool
def get_admet_predictions(csv_file_path):
    """
    This tool predicts ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) 
    properties by interacting with the DeepPK web API. 

    Arguments:
    - csv_file_path (str): Path to a CSV file containing SMILES strings with a header column named 'SMILES'.

    Returns:
    - dict: A dictionary containing ADMET predictions, or an error message if the job fails or times out.
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
                # save json
                # with open('property/admet.json', 'w') as f:
                #     json.dump(result_json, f)
                df = pd.DataFrame(result_json).T
                df.to_csv('property/admet.csv', index=False)
                # df = pd.DataFrame.from_dict(result_json['0'], orient='index')
                # df = df.transpose()
                # df.to_csv('property/admet.csv', index=False)

            return "ADMET predictions saved to property/admet.csv"
        except Exception as e:
            return {"error": f"Polling failed: {e}"}



# @tool
# def read_dataset_columns(csv_path: str):
#     """
#     Reads the property columns of the dataset in CSV format.

#     Args:
#         csv_path: Path to the input CSV file.

#     Returns:
#         list: List of column names in the dataset.
#     """
#     try:
#         # Read only the column names
#         df = pd.read_csv(csv_path, nrows=0)
#         columns = df.columns.tolist()
#         return columns

#     except Exception as e:
#         print(f"Error reading the dataset columns: {str(e)}")
#         return []
    
# write a function to get this dictionary
@tool
def read_dataset(proerty_file_path: str):
    """
    Get the dictionary of the first row of the dataframe to understand the dataset.
    Make sure the file path format is valid.
    Args:
        proerty_file_path: Path to the input CSV file.

    Returns:
        dict: Dictionary representation of the random row in the dataset.
    """
    try:
        
        df = pd.read_csv(proerty_file_path)
        idx = random.randint(0, len(df)-1)
        ex_dict = df.iloc[idx].to_dict()
        return ex_dict
    except Exception as e:
        print(f"Error getting the dictionary: {str(e)}")
        return {}
