def get_retriever(vectordb, k: int = 5):
    return vectordb.as_retriever(search_kwargs={"k": k})
