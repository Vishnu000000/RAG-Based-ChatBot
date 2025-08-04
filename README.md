# RAG-based Chatbot with LangChain and OpenAI

This project is a command-line chatbot that uses the **Retrieval-Augmented Generation (RAG)** architecture to answer questions based on a custom knowledge base.

It is a powerful demonstration of modern AI application development, showcasing how to build systems that can reason about specific information provided to them, mitigating the risk of "hallucination" common in large language models (LLMs).

---

## Architecture: How It Works

The RAG architecture combines the power of a pre-trained LLM with a specific information retrieval component.

1.  **Setup (`setup.py`):**
    -   A source document (`knowledge_base.txt`) is loaded.
    -   The document is split into smaller, manageable chunks.
    -   A local embedding model (`sentence-transformers`) converts each chunk into a numerical vector representation.
    -   These vectors are stored and indexed in a **FAISS vector store**, which is saved locally. This is our searchable knowledge base.

2.  **Chat (`chatbot.py`):**
    -   A user asks a question.
    -   The user's question is converted into an embedding vector.
    -   The FAISS vector store is searched to find the document chunks with embeddings most similar to the question's embedding. These are the most **relevant context**.
    -   The original question, along with the retrieved context, is sent to an LLM (OpenAI's GPT-3.5).
    -   The LLM generates a final answer, using the provided context to ensure the answer is factually grounded in the source document.

---

## Core Concepts Demonstrated

-   **Advanced AI (Block 8):**
    -   **Retrieval-Augmented Generation (RAG):** The core architecture of the project.
    -   **LLM Integration:** Using LangChain to interact with OpenAI's API.
    -   **Embeddings & Vector Stores:** Understanding how to represent text numerically and perform efficient similarity searches using Sentence-Transformers and FAISS.
-   **MLOps (Block 8):**
    -   Demonstrates a key pattern for productionizing LLM applications by separating the one-time indexing step (`setup.py`) from the real-time serving application (`chatbot.py`).

---

## How to Run

### Prerequisites

1.  **Python 3.8+**
2.  An **OpenAI API Key**.

### 1. Setup Your Environment

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/rag-chatbot.git](https://github.com/YOUR_USERNAME/rag-chatbot.git)
cd rag-chatbot

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API Key
# Replace 'your_key_here' with your actual key
export OPENAI_API_KEY='your_key_here'
# On Windows Command Prompt, use: set OPENAI_API_KEY=your_key_here
# On Windows PowerShell, use: $env:OPENAI_API_KEY="your_key_here"
```

### 2. Create the Knowledge Base

Run the setup script. This will read `knowledge_base.txt`, create the embeddings, and save the FAISS index. You only need to do this once (or whenever you update `knowledge_base.txt`).

```bash
python setup.py
```
This might take a moment as it downloads the embedding model for the first time.

### 3. Run the Chatbot

Now you can start the chatbot and ask it questions about the content in `knowledge_base.txt`.
