# Step 1: Import necessary libraries
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 2: Load and chunk the pdf document

loader = PyPDFLoader('report.pdf')
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

chunks = splitter.split_documents(pages)

print(f"Chunks ready: {len(chunks)}")
print("Loading embedding model... (first time will download ~90MB, be patient)")

# Step 3: Load a free local embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded. Creating vector store...")

# Step 4: convert chunks to vectors and store in FAISS
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store created successfully!")

# Step 5: Save the  vector store to disk for later use
vector_store.save_local("faiss_index")
print("Vector store saved to disk as 'faiss_index'")
