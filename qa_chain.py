from dotenv import load_dotenv
load_dotenv()

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Step 1 - Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2 - Load saved FAISS index
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Vector store loaded!")

# Step 3 - Define prompt template
prompt_template = PromptTemplate.from_template("""
Use the following context from an annual report to answer the question.
Be specific and use numbers where available.
If the answer is not in the context, say "I could not find this information in the report."

Context:
{context}

Question: {question}

Answer:
""")

# Step 4 - Load LLM via Groq (free)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 5 - Build retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# Step 6 - Helper to format retrieved chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 7 - Build RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

print("RAG chain ready! Asking question...\n")

# Step 8 - Ask a question
question = "What are the major business, financial and operational risk factors mentioned in the report?"
answer = rag_chain.invoke(question)

print(f"Question: {question}")
print(f"\nAnswer: {answer}")