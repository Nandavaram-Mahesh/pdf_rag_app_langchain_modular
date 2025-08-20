from langchain.prompts import ChatPromptTemplate
def chat_prompt_template():
    template = """Answer the question based only on the following context:
                {context}

                Question: {question}
                """ 
    return ChatPromptTemplate.from_template(template)