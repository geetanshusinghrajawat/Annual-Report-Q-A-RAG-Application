from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# step 1: load the same embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#step 2: load the saved vector store from disk
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True # Required by newer langchain versions
)
print("Vector store loaded successfully!")

# Step 3: Ask a quetion and retrieve relevant chunks from the vector store
question = "What is the total revenue?"
results = vector_store.similarity_search(question, k=3) # k retrives the top most relevant chunks which is 3 here

print(f"\nTop 3 chunks retrieved for: '{question}'")
print("=" * 60)

for i, doc in enumerate(results):
    print(f"\nChunk {i+1}:")
    print(doc.page_content)
    print("-" * 60)