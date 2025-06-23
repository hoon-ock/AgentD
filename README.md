# agentD

A package for drug discovery AI agents and tools.

---

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/hoon-ock/llm-dd.git
    cd llm_dd
    ```

2. **Install dependencies in editable mode:**
    ```sh
    pip install -e .
    ```
    Or, to install all dependencies directly:
    ```sh
    pip install -r requirements.txt
    ```

3. **Install REINVENT4 (required for some tools):**
    ```sh
    git clone https://github.com/MolecularAI/Reinvent.git
    cd Reinvent
    python install.py --help
    python install.py cu124  # or rocm6.2.4, cpu, mac, etc.
    ```

---

## Configuration

- **API Keys:**  
  After installation, copy `configs/secret_keys.py.example` to [secret_keys.py](http://_vscodecontentref_/0) and fill in your Serper API key and OpenAI API key:
    ```python
    # configs/secret_keys.py
    serper_api_key = "YOUR_SERPER_API_KEY"
    openai_api_key = "YOUR_OPENAI_API_KEY"
    ```

- **Global Variables:**  
  The file [tool_globals.py](http://_vscodecontentref_/1) contains global variables used by the tools. You can edit this file to adjust default behaviors and settings.

---

## Example Notebooks

Example Jupyter notebooks demonstrating the main workflows are provided in the [test_case](http://_vscodecontentref_/2) directory:

1. `1. extraction.ipynb` – Data extraction workflow
2. `2. qna.ipynb` – Question answering with LLM agent
3. `3. pooling.ipynb` – Pooling and data aggregation
4. `4. prediction.ipynb` – Affinity prediction
5. `5. refinement.ipynb` – Molecule refinement
6. `6. generation.ipynb` – Molecule generation

You can run these notebooks step-by-step to see how to use the package for various drug discovery tasks.

---

## License

This project is licensed under the MIT License.

---

## Notes

- Make sure to set up your API keys before running any LLM agent notebooks.
- For any additional dependencies (e.g., REINVENT4), follow the instructions above.
- If you encounter missing package errors, check that all dependencies in [requirements.txt](http://_vscodecontentref_/3) are installed.

---

## Contact

For questions, suggestions, or support, please contact:  
**Hoon Ock**  
Email: [jock@andrew.cmu.edu](mailto:jock@andrew.cmu.edu)