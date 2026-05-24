import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.title("📊 Annual Report Q&A")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def build_vector_store(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_documents(pages)
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.unlink(tmp_path)
    return vector_store

uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        vector_store = build_vector_store(uploaded_file.read())
    st.success("PDF processed! Ask your question below.")

    question = st.text_input("Ask a question about the report:")

    if question:
        prompt_template = PromptTemplate.from_template("""
Use the following context from an annual report to answer the question.
Be specific and use numbers where available.
If the answer is not in the context, say "I could not find this information in the report."

Context:
{context}

Question: {question}

Answer:
""")

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 6})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            answer = chain.invoke(question)

        st.markdown("### Answer")
        st.write(answer)