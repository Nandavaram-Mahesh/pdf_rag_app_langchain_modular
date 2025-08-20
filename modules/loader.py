from typing import List
from langchain.docstore.document import Document 
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(file_path:List[str])-> List[Document]:
    docs = []
    
    loader = PyMuPDFLoader(file_path[0])
    # Lazy Loading
    for doc in loader.lazy_load():
        docs.append(doc)
    return docs
