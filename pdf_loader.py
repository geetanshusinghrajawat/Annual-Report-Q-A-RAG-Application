
# Step 1: Import necessary libraries

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 2: Load the PDF document
loader = PyPDFLoader('report.pdf')
pages = loader.load()

print(f"Total pages Loaded: {len(pages)}")
print(f"\nSample text from page 1:\n{pages[0].page_content[:500]}") # Print the first 500 characters of the first page

# Step 3: Split the text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = splitter.split_documents(pages)

print(f"\n Total chunks created: {len(chunks)}")
print(f"\n Sample Chunk 1: {chunks[0].page_content}")
print(f"\n Sample Chunk 2: {chunks[1].page_content}")