from langchain_community.vectorstores import FAISS

def build_vectorstore(docs, embeddings, index_dir: str):
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    vectordb.save_local(index_dir)
    return vectordb

def load_vectorstore(index_dir: str, embeddings):
    return FAISS.load_local(index_dir, embeddings,allow_dangerous_deserialization=True)
