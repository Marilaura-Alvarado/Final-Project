#!/usr/bin/env python
# coding: utf-8

import os
import json
import streamlit as st
from pathlib import Path

from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS



BASE_DIR = Path(__file__).resolve().parents[1]



st.set_page_config(
    page_title="AI search with chat",
    page_icon="💬",
    layout="wide"
)


st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0b1f3a 0%, #0e2a52 100%);
            color: #e6edf3;
        }
        h1, h2, h3 {
            color: #7fb0ff;
        }
        section[data-testid="stSidebar"] {
            background-color: #081a33;
        }
        textarea {
            background-color: #081a33 !important;
            color: #e6edf3 !important;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #1f4fd8;
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover {
            background-color: #163fa8;
        }
        .block {
            background-color: #0f2f5c;
            border-left: 5px solid #7fb0ff;
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.header("Chat-bot with LLM")
st.header("AI-bot for RAG based search", divider="")

st.markdown(
    """
    This application implements a Retrieval-Augmented Generation (RAG) pipeline.
    It retrieves relevant information from PDF documents and uses an LLM
    to generate grounded answers.
    """
)


def read_json(filename):
    file_path = BASE_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def initialize_faiss_vectorstore():

    API_CREDS = read_json("apicreds2.json")
    APP_CONFIG = read_json("config.json")

   
    DATA_PATH = Path(APP_CONFIG["imgs_path"]) / "rag"

    FAISS_INDEX_PATH = BASE_DIR / "faiss_index"

    embeddings = YandexGPTEmbeddings(
        api_key=API_CREDS["api_key"],
        folder_id=API_CREDS["folder_id"]
    )

    if FAISS_INDEX_PATH.exists():
        with st.spinner("Loading FAISS index..."):
            vectorstore = FAISS.load_local(
                str(FAISS_INDEX_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
        st.success("FAISS index loaded")
        return vectorstore, API_CREDS

    if not DATA_PATH.exists():
        st.warning("Knowledge base directory not found.")
        st.code(str(DATA_PATH))
        st.info("Create this folder and add PDF files to enable search.")
        return None, API_CREDS

    loader = DirectoryLoader(
        str(DATA_PATH),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

    if not documents:
        st.warning("No PDF documents found in the knowledge base.")
        return None, API_CREDS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20000,
        chunk_overlap=100
    )
    splits = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_PATH))

    st.success("FAISS index created and saved")
    return vectorstore, API_CREDS


def get_rag_chain(vectorstore, template, temperature, k_max, api_creds):

    retriever = vectorstore.as_retriever(search_kwargs={"k": k_max})
    prompt = PromptTemplate.from_template(template)

    llm = ChatYandexGPT(
        api_key=api_creds["api_key"],
        folder_id=api_creds["folder_id"],
        temperature=temperature
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


if "vectorstore" not in st.session_state:
    result = initialize_faiss_vectorstore()
    if result[0] is None:
        st.stop()
    st.session_state.vectorstore, st.session_state.api_creds = result



default_instruction = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Always say "thanks for asking!" at the end.

Context: {context}
Question: {question}
Answer:
"""

st.markdown('<div class="block">', unsafe_allow_html=True)
template = st.text_area("Prompt template", default_instruction, height=200)
temperature = st.slider("Creativity level", 0.0, 1.0, 0.0, 0.1)
k_max = st.slider("Top-k documents", 1, 5, 3)
st.markdown('</div>', unsafe_allow_html=True)



rag_chain = get_rag_chain(
    st.session_state.vectorstore,
    template,
    temperature,
    k_max,
    st.session_state.api_creds
)



st.write("### Ask the chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Type your question"):
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    answer = rag_chain.invoke(query)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
