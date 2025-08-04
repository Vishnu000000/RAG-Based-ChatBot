# File: setup.py
# This script is responsible for setting up our knowledge base.
# It performs the one-time task of reading a text file, splitting it
# into manageable chunks, creating vector embeddings for each chunk,
# and saving them into a FAISS vector store. This vector store acts
# as our indexed, searchable knowledge base.

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_knowledge_base():
    """
    Reads the knowledge base file, creates embeddings, and saves them
    to a local FAISS vector store.
    """
    # Step 1: Load the source document
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            source_text = f.read()
        print("Knowledge base file loaded.")
    except FileNotFoundError:
        print("Error: 'knowledge_base.txt' not found.")
        print("Please create this file and add some text to it.")
        return

    # Step 2: Split the document into smaller chunks
    # This is important because LLMs have a limited context window.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # The size of each chunk in characters
        chunk_overlap=200, # The number of characters to overlap between chunks
        length_function=len
    )
    text_chunks = text_splitter.split_text(source_text)
    print(f"Text split into {len(text_chunks)} chunks.")

    # Step 3: Create embeddings for the text chunks
    # We use a model from HuggingFace that runs locally.
    # "all-MiniLM-L6-v2" is a popular and efficient model.
    print("Creating embeddings... (This might take a moment)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Step 4: Create a FAISS vector store from the chunks and embeddings
    # FAISS is a library for efficient similarity search on dense vectors.
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("Vector store created.")

    # Step 5: Save the vector store locally
    # This allows us to load it later without re-processing the documents.
    vector_store.save_local("faiss_index")
    print("FAISS index saved locally. Setup is complete.")
    print("You can now run chatbot.py")

if __name__ == "__main__":
    create_knowledge_base()
