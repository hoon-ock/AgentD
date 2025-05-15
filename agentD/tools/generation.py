from langchain.tools import tool
import json, yaml, os, subprocess
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI  # or your preferred provider (e.g., Ollama, HuggingFaceHub)
from langchain.chains import LLMChain
os.environ["PATH"] += os.pathsep + "/home/hoon/dd-agent/alphafold/localcolabfold/colabfold-conda/bin"

REINVENT_PATH = "/home/hoon/dd-agent/REINVENT4"


@tool
def update_reinvent_config(model_type: str = "Reinvent"): # 
    """
    Uses an LLM to update relative paths in a TOML config file. This returns the path to the updated config file.

    Args:
        model_type (str): The model type to use for updating paths. Default is "Reinvent". This can be "Reinvent", "LibInvent", "LinkInvent", "Mol2Mol", or "Pepinvent".
                       Currently, only "Reinvent" and "Mol2Mol" are supported in the agentD framework.
    """
    # Fixed input TOML path
    prefix = REINVENT_PATH
    #input_path = "/home/hoon/dd-agent/REINVENT4/configs/toml/staged_learning.toml" #"configs/staged_learning.toml"
    input_path = os.path.join(prefix, "configs", "toml", "sampling.toml")
    # Read original TOML content
    with open(input_path, "r") as f:
        toml_text = f.read()

    # Create prompt template
    prompt_template = PromptTemplate(
    input_variables=["toml", "prefix", "model_type"],
    template="""
Your task is to update the provided TOML content as follows:

1. Uncomment the paths in the specified {model_type} section.
2. Comment out paths in all other model types, "Reinvent", "LibInvent", "LinkInvent", "Mol2Mol", and "Pepinvent".
3. Prepend the path "{prefix}" to model file paths in the TOML file.
4. Change the output_file path to "results/{model_type}_sampling.csv".
5. If there is smiles_file, prepend the path "configs" to the smiles_file path.
6. Do NOT modify any other commented lines or sections.
7. Return ONLY the modified TOML content — no additional text, explanations, or formatting.

Strictly output the updated TOML content only, without any ```toml``` prefix or any other surrounding text.

TOML:
{toml}
"""
)

    # Initialize LLM (swap this if you're using Ollama, HuggingFace, etc.)
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0.0)
    # llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)
    chain = LLMChain(prompt=prompt_template, llm=llm)

    # Get updated TOML content
    updated_toml = chain.run(toml=toml_text, prefix=prefix, model_type=model_type)
    
    # Save to output file
    output_path = f'configs/{model_type}.toml'#input_path.split("/")[-1]
    with open(output_path, "w") as f:
        f.write(updated_toml)

    print(f"Updated TOML file saved to: {output_path}")
    return output_path

@tool
def run_reinvent(config_file: str):
    """
    Run the REINVENT command with the specified log and configuration files to generate a pool of candidate molecules.
    Args:
        config_file (str): The path to the configuration .toml file.
    """
    #log_file="staged_learning.log", 
    #config_file=os.path.join(REINVENT_PATH, "configs/toml/staged_learning.toml")
    log_file = config_file.replace(".toml", ".log")
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