from langchain_openai import OpenAIEmbeddings

def get_embedder(api_key: str, model: str):
    return OpenAIEmbeddings(openai_api_key=api_key, model=model)
