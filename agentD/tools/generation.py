from langchain.tools import tool
import json, yaml, os, subprocess
import subprocess
os.environ["PATH"] += os.pathsep + "/home/hoon/dd-agent/alphafold/localcolabfold/colabfold-conda/bin"

REINVENT_PATH = "/home/hoon/dd-agent/REINVENT4"


@tool
def generate_pool(prompt: str):
    """
    Run the REINVENT command with specified log and config files.

    Args:
        log_file (str): The log file name to save the output.
        config_file (str): The path to the configuration .toml file.
    """
    log_file="staged_learning.log", 
    config_file=os.path.join(REINVENT_PATH, "configs/toml/staged_learning.toml")
    command = [
        "reinvent",
        "-l", log_file,
        config_file
    ]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        return "REINVENT execution completed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error occurred while running REINVENT: {e}"
    except FileNotFoundError:
        return "Error: 'reinvent' command not found. Make sure REINVENT is installed and accessible in your PATH."

@tool
def run_NAMDagent():
    pass

@tool
def generate_structure(data: str):
    """
    Create a protein-ligand complex structure using Boltz from a given protein sequence and ligand SMILES.
    This function first creates a YAML config file for Boltz and then runs the structure prediction.
    To run this tool, you need to provide a dictionary with "sequence" and "smiles" keys.
    The sequence should be a string representing the protein sequence, and smiles should be a string
    representing the ligand SMILES string.

    Args:
        data: valid JSON Dictionary with "sequence" (str) and "smiles" (str).
    """
    # Step 1: Parse the input data
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return "Error: Invalid input format. Expected JSON with 'sequence' and 'smiles'."

    protein_seq = data.get("sequence")
    ligand_smiles = data.get("smiles")
    # Step 2: Create the YAML config file
    config_data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "[A]", "sequence": protein_seq}},
            {"ligand": {"id": "[B]", "smiles": ligand_smiles}},
        ],
    }
    yaml_filename = "config.yaml" # os.path.join(GEN_PATH, "config.yaml") #
    with open(yaml_filename, "w") as file:
        yaml.dump(config_data, file, default_flow_style=False)

    print(f"YAML file created: {yaml_filename}")

    # Step 3: Run Boltz to predict the protein-ligand complex
    boltz_command = f"boltz predict {yaml_filename} --use_msa_server --accelerator gpu --num_workers 20 "
    return "Config file created. Now the user can run Boltz with the command: " + boltz_command
    # try:
    #     torch.cuda.empty_cache() # Clear GPU cache before running Boltzf
    #     subprocess.run(boltz_command, shell=True, check=True)
    #     print("Boltz structure prediction completed.")
    #     torch.cuda.empty_cache()  # Clear GPU cache after running Boltz
    # except subprocess.CalledProcessError as e:
    #     print(f"Error running Boltz: {e}")
    #     return None
    
    #return "predicted_structure.pdb"  # Change this based on Boltz output