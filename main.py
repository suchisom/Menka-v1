import streamlit as st
import os
import time
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# UPDATED: Import both Chat and Embedding models from Mistral
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings




MISTRAL_MODEL_NAME = "open-mistral-7b"
# UPDATED: The dimension for Mistral's embedding model is 1024
PINECONE_INDEX_DIMENSION = 1024


    # For Streamlit Community Cloud, get the key from st.secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
mistral_api_key = st.secrets["MISTRAL_API_KEY"]


#load_dotenv()
#mistral_api_key = os.getenv("MISTRAL_API_KEY")
#pinecone_api_key = os.getenv("PINECONE_API_KEY")

# ui-
st.set_page_config(page_title="Menka-Rag", layout="wide")

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

# Custom CSS to hide/show sidebar
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        display: {'none' if st.session_state.sidebar_state == 'collapsed' else 'block'};
    }}
    </style>
    """, unsafe_allow_html=True)


st.markdown("""
## Menka: Docs to Info

### HOW TO USE
1.  **Upload docs**: Open the sidebar and upload your sources.
2.  **Submit & process**: You will have to wait a while.
3.  **Ask a Question**: The LLM is not really smart but it will get u through ;) .
4. For feedback and suggestions 
""")
st.write("You can send feedback and suggestions.")
# This opens email app to compose a new email to me
st.link_button("Email Me Directly", "mailto:suzzzylabs@gmail.com")

# Add animated background

index_name = st.text_input(
        "Pinecone Index Name:", 
        key="pinecone_index_name_input", 
        value=os.getenv("PINECONE_INDEX_NAME", "mistral-rag-index")
    )  

# Funks
@st.cache_resource
def get_pinecone_client(api_key):
    """Initializes and returns a Pinecone client."""
    if not api_key:
        st.error("Pinecone API Key is missing.")
        return None
    return Pinecone(api_key=api_key)

def clean_text(text):
    """Clean text by removing problematic Unicode characters."""
    import re
    
    # Remove surrogate characters and other problematic Unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Remove non-printable characters except newlines, tabs, and spaces
    text = re.sub(r'[^\x20-\x7E\n\t\r]', ' ', text)
    
    # Replace multiple whitespaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                # Clean the extracted text
                cleaned_text = clean_text(page_text)
                text += cleaned_text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
    return text

def get_text_chunks(raw_text):
    # Additional cleaning before chunking
    cleaned_text = clean_text(raw_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(cleaned_text)
    
    # Clean each chunk individually to be extra safe
    cleaned_chunks = [clean_text(chunk) for chunk in chunks if chunk.strip()]
    return cleaned_chunks

def get_vector_store_pinecone(text_chunks, index_name):
    """Creates embeddings and upserts documents into a Pinecone vector index."""
    pc = get_pinecone_client(pinecone_api_key)
    if not all([pc, mistral_api_key, index_name]):
        st.error("API Keys, Index Name missing, or Pinecone client failed to initialize.")
        return

    try:
        # Use MistralAIEmbeddings
        embeddings = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
        documents = [Document(page_content=t) for t in text_chunks]
        
        index_exists = index_name in pc.list_indexes().names()
        
        if not index_name:
            st.info(f"Creating new Pinecone index: '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=PINECONE_INDEX_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            st.success(f"Index '{index_name}' created and ready.")
        
        else:
           st.info(f"Using existing index: '{index_name}'") 
        
        if index_exists:
           vector_store = PineconeVectorStore.from_existing_index(
           index_name=index_name,
           embedding=embeddings
                             )
    
        else:
          PineconeVectorStore.from_documents(
            documents=documents, 
            embedding=embeddings, 
            index_name=index_name,
            pinecone_api_key=pinecone_api_key
        )
        st.success("✅ Documents successfully processed and stored in Pinecone!")
        
    except Exception as e:
        st.error(f"An error occurred during Pinecone operations: {e}")

def get_conversational_chain():
    """Configures the Mistral QA chain."""
    prompt_template = """
    You are an expert assistant. Answer the question in detail based on the provided context.
    If the answer is not found in the context, clearly state, "The answer is not available in the provided documents."
    Do not use any external knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatMistralAI(model=MISTRAL_MODEL_NAME, temperature=0.3, mistral_api_key=mistral_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

        
def handle_user_input(user_question, index_name):
    """
    Performs similarity search, runs the QA chain, and returns the response.
    """
    # This check is now more comprehensive
    if not all([pinecone_api_key, mistral_api_key, index_name]):
        return "Error: API keys or Index Name are missing. Please configure them in the sidebar."

    try:
        embeddings = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
        
        vector_store = PineconeVectorStore.from_existing_index(
               index_name=index_name, 
            embedding=embeddings
                           )
        
        docs = vector_store.similarity_search(user_question, k=5)

        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            # MODIFICATION: Instead of printing, we return the output text
            return response["output_text"]
        else:
            return "Error: Failed to initialize the conversational chain."

    except Exception as e:
        # Check for a specific common error: index not found
        if "not found" in str(e).lower() and "index" in str(e).lower():
             return (f"Error: The Pinecone index '{index_name}' was not found. "
                     "Please upload and process your documents first to create it.")
        return f"An error occurred during the query: {e}"

def main():
    # Sidebar for document processing (this part remains the same)
    with st.sidebar:
        st.subheader("Document Uploader")
        pdf_docs = st.file_uploader("Upload PDF files and click 'Process'", accept_multiple_files=True, key="pdf_uploader")
        
        if st.button("Submit & Process", key="process_button"):
            if not all([pinecone_api_key, mistral_api_key, index_name]):
                st.warning("Please ensure all API keys and an index name are provided.")
            elif pdf_docs:
                with st.spinner("Processing documents... This may take a moment."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store_pinecone(text_chunks, index_name)
                    else:
                        st.warning("Could not extract text from the provided PDFs.")
            else:
                st.warning("Please upload at least one PDF file.")

    #  Main Chat
    st.header("Chat with Your Documents")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Hello i'm here to assist, enjoy your stay."
        }]

    # Display past messages from the history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_user_input(prompt, index_name)
                st.markdown(response)
        
        # Add assistant respo to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


