import os
from typing import List, Dict
from .loader import load_pdfs
from .splitter import split_documents
from .embedder import get_embedder
from .vectorstore import build_vectorstore, load_vectorstore
from .retriever import get_retriever
from .llm import get_llm
from .chat_prompt_template import chat_prompt_template
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def load_langsmith_cred(langchain_api_key):
    if langchain_api_key:
        # Step 2: Set env variables only after key is entered
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGSMITH_PROJECT"] = "LANGCHAIN_MODULAR_RAG_PDF"
    
def build_index(pdf_path: List[str], index_dir: str, api_key: str, embed_model: str, chunk_size: int, chunk_overlap: int) -> int:
    docs = load_pdfs(pdf_path)
    splits = split_documents(docs, chunk_size, chunk_overlap)
    embedder = get_embedder(api_key, embed_model)
    build_vectorstore(splits, embedder, index_dir)
    return len(splits)

def answer_query(query: str, index_dir: str, api_key: str, embed_model: str, chat_model: str, k: int = 5) -> Dict:
    embedder = get_embedder(api_key, embed_model)
    vectordb = load_vectorstore(index_dir, embedder)
    retriever = get_retriever(vectordb, k=k)
    prompt = chat_prompt_template()
    llm = get_llm(api_key, chat_model)
    
    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text
    
    rag_chain =({"context": retriever| RunnableLambda(format_docs), "question": RunnablePassthrough()}
                |prompt
                |llm
                |StrOutputParser()
                )

    response = rag_chain.invoke(query)
    
    return {'response': response,}
