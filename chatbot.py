# File: chatbot.py
# This is the main application for our RAG chatbot.
# It loads the pre-built vector store and sets up a conversational
# chain using LangChain to answer questions based on the knowledge base.

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# IMPORTANT: You need to set your OpenAI API key as an environment variable.
# For example, in your terminal: export OPENAI_API_KEY='your_key_here'
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key to run this chatbot.")
    exit()

def run_chatbot():
    """
    Initializes and runs the RAG chatbot loop.
    """
    # --- Initialization ---
    try:
        # Load the embeddings model (must be the same as used in setup.py)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load the FAISS vector store we created in setup.py
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
    except Exception as e:
        print(f"Error loading the FAISS index: {e}")
        print("Please make sure you have run 'python setup.py' first to create the index.")
        return

    # Create a retriever from the vector store.
    # The retriever's job is to find the most relevant document chunks for a given query.
    retriever = vector_store.as_retriever()

    # Initialize the Language Model (LLM) - in this case, OpenAI's GPT-3.5 Turbo
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    # Create the Conversational Retrieval Chain. This is the core of the RAG application.
    # It combines the retriever and the LLM to generate answers.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    # --- Chat Loop ---
    chat_history = []
    print("\nChatbot is ready! Type 'exit' to end the session.")
    print("Ask a question about the content in your knowledge base.")
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() == 'exit':
                print("Chatbot session ended. Goodbye!")
                break
            
            # The chain takes the user's question and the chat history as input.
            result = conversation_chain.invoke({"question": query, "chat_history": chat_history})
            
            # The answer from the LLM is in the 'answer' key of the result.
            answer = result['answer']
            print(f"\nBot: {answer}")
            
            # Update the chat history to provide context for future questions.
            chat_history.append((query, answer))

        except KeyboardInterrupt:
            print("\nChatbot session ended. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    run_chatbot()
