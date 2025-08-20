# PDF RAG App (LangChain + Streamlit, Modular)

A modular Retrieval-Augmented Generation (RAG) demo for **PDFs**, built with **LangChain**, **Streamlit**, and **OpenAI**.

## Structure
```
pdf_rag_app_langchain_modular/
├── app.py
├── config.py
├── modules/
│   ├── __init__.py
│   ├── loader.py
│   ├── splitter.py
│   ├── embedder.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # Windows: setx OPENAI_API_KEY "sk-..."
streamlit run app.py
```
## Deployed App
```
https://nandavaram-mahesh-pdf-rag-app-langchain-modular-app-h80nu2.streamlit.app/
```
