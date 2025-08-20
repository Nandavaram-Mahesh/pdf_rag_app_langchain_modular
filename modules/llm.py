from langchain_openai import ChatOpenAI

def get_llm(api_key: str, model: str, temperature: float = 0):
    return ChatOpenAI(openai_api_key=api_key, model=model, temperature=temperature,max_tokens=1000)


