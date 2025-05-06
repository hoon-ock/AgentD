from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from chembl_webresource_client.new_client import new_client
import requests
import json
import random
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

NUM_IDS = 1
SIMILARITY_THRESHOLD = 40
SAMPLING_SIZE = 5


@tool
def search(query):
    '''
    This tool performs a search on Google using SerpAPI and returns relevant information for the query.
    It is useful for retrieving information from the web or when you need to look up current data.
    '''
    search = GoogleSerperAPIWrapper(engine='google', serpapi_api_key=os.getenv("SERPER_API_KEY"))
    results = search.run(query)
    return results

@tool
def get_uniprot_ids(protein_name: str):
    """
    Searches for UniProt IDs based on a given protein name and returns the IDs.
    """
    import requests

    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"{protein_name} AND organism_id:9606"  # Ensure the query is formatted properly
    # size  = 5
    params = {
        "query": query,
        "format": "json",
        "size": NUM_IDS  # Number of results to retrieve
    }

    #print(f"🔍 Sending API request: {base_url}?query={query}&format=json&size={size}")  # Debugging output

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for HTTP failures
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            return data['results'][0]['primaryAccession'] #[entry["primaryAccession"] for entry in data["results"]][0]  # Return the first UniProt ID
        else:
            return f"No UniProt IDs found for '{protein_name}'."

    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    
@tool
def fetch_uniprot_fasta(uniprot_id):
    """
    Fetch the FASTA sequence using the UniProt ID.
    """
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(fasta_url)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch FASTA for {uniprot_id}. Status Code: {response.status_code}")
        return None
    
@tool
def get_drug_smiles(drug_name: str):
    """
    Fetch the drug molecule SMILES of a given drug name from ChEMBL.
    """
    molecule = new_client.molecule
    mols = molecule.filter(pref_name__iexact=drug_name)

    matching_mols = defaultdict(list)
    
    for mol in mols:
        if 'molecule_structures' in mol and mol['molecule_structures']:
            matching_mols[mol['molecule_chembl_id']].append(mol['molecule_structures']['canonical_smiles'])

    if not matching_mols:
        print(f"No SMILES found for {drug_name}.")
        return None, None
    
    # Get the first match as the anchor molecule
    anchor_chembl_id = list(matching_mols.keys())[0]
    anchor_smiles = matching_mols[anchor_chembl_id][0]
    
    return anchor_smiles

@tool
def get_seed_molecule(anchor_smiles: str):
    """
    Retrieve seed molecules from ChEMBL for further optimization based on the given SMILES.
    
    The returned molecule is structurally distinct from the input and selected from candidates 
    with similarity below a defined threshold.
    """
    similarity = new_client.similarity
    sim_mols = similarity.filter(smiles=anchor_smiles, similarity=SIMILARITY_THRESHOLD).only(
        ['molecule_chembl_id', 'similarity', 'molecule_structures']
    )

    non_similar_mols = []
    for mol in sim_mols:
        if ('molecule_structures' in mol and float(mol['similarity']) < SIMILARITY_THRESHOLD+5):
            
            non_similar_mols.append(mol['molecule_structures']['canonical_smiles'])

    if not non_similar_mols:
        return None
    random.shuffle(non_similar_mols)
    
    return non_similar_mols[:SAMPLING_SIZE]
    # return random.choice(non_similar_mols)

@tool
def write_file(input_str: str) -> str:
    """
    Write content to a file. Input should be a JSON string.
    """
    
    try:
        data = json.loads(input_str)
        filename = data["name"]
  
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=4))
        return f"Saved to {filename}"
    except Exception as e:
        return f"Failed to write file: {e}"