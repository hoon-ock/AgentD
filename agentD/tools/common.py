import json
from langchain.tools import tool
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