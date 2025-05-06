from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import tool
import os, requests, time, sys

cwd = os.getcwd()
home_dir = os.path.dirname(os.path.dirname(cwd))
sys.path.append(home_dir)
from agentD.prompts import RETRIEVAL_QA
from agentD.utils import download_pdf, RetrievalQABypassTokenLimit

MAX_PAPERS = 5 # Number of papers to download for molecule optimization guideline
PAPER_DIR = "/home/hoon/dd-agent/llm_dd/agentD/papers"
EMBEDDING_MODEL = 'text-embedding-3-large' 
# from https://mehradans92.github.io/dZiner/peptide-hemolytic.html
# need to define Embedding_model, RetrievalQA_prompt
@tool
def domain_knowledge(path: str):
    '''
    This tool derives design guidelines for drug molecule with higher binding affinity by looking through research papers.
    This tool takes a path toward the directory where the papers are stored and searches for relevant papers.
    It also includes information on the paper citation or DOI.
    '''
    # check if the files exist in the directory
    if not os.path.exists(PAPER_DIR):
        return "No papers found in the directory. Use generic design guidelines instead."

    guide_lines = []
    # iterate over the downloaded papers
    for paper_file in os.listdir(PAPER_DIR):
        # check if the file is a PDF
        if not paper_file.endswith('.pdf'):
            continue
        # construct the full file path
        paper_file = os.path.join(PAPER_DIR, paper_file)
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50)
        pages = PyPDFLoader(paper_file).load_and_split()
        sliced_pages = text_splitter.split_documents(pages)
        faiss_vectorstore = FAISS.from_documents(sliced_pages, OpenAIEmbeddings(model=EMBEDDING_MODEL))
        
        llm=ChatOpenAI(
                        model_name='gpt-4o',
                        temperature=0.3,
                        )
        g = RetrievalQABypassTokenLimit(faiss_vectorstore, RETRIEVAL_QA, llm)
        guide_lines.append(g)
    return " ".join(guide_lines)

@tool
def download_relevant_papers(query: str):
    """
    Searches for and downloads relevant academic papers related to a given research query.
    This function uses the Semantic Scholar API to search for papers and download them if they are open access.
    Query should be constructed in a way that it can be used to search for relevant papers.
    **Example Usage:**
        download_relevant_papers("drug molecule for <<target protein>>")
    """
    
    print(f"Searching for papers on: {query}")

    papers_downloaded = []

    # Step 1: Search Semantic Scholar
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": "P5pZs85BTC4MGCCNIQaDPaO2ktEIVZI08JKKBTox"}  # Replace with a valid API key

    params = {
        "query": query,
        "fields": "title,url,abstract,isOpenAccess,openAccessPdf",
        "limit": MAX_PAPERS
    }

    response = requests.get(SEMANTIC_SCHOLAR_API, headers=headers, params=params)

    if response.status_code == 200:
        results = response.json().get("data", [])
        for paper in results:
            title = paper.get("title", "Untitled")  # Default to 'Untitled' if title is missing
            open_access_pdf = paper.get("openAccessPdf")  # Get the dictionary (could be None)
  
            if isinstance(open_access_pdf, dict):  # Ensure it's a dictionary
                pdf_url = open_access_pdf.get("url")
                if pdf_url:
                    file_path = download_pdf(pdf_url, title, PAPER_DIR)
                    if file_path:
                        papers_downloaded.append(file_path)
                    time.sleep(2)
                # papers_downloaded.append(file_path)
    else:
        print("Error fetching papers from Semantic Scholar.")

    return papers_downloaded