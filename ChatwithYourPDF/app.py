import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.schema import Document
import os
import requests
import logging
import json

# Set up logging
logger = logging.getLogger(__name__)

# Function to set up the sidebar
def setup_sidebar():
    with st.sidebar:
        st.title('📚💬 PDF Chat Assistant')
        st.markdown('''
        ## About
        This app allows you to chat with your PDF documents using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) language models
        - [Llumo](https://www.llumo.ai/) for text compression
       
        Upload a PDF and start chatting to get answers instantly!
        ''')
        add_vertical_space(5)
        st.write('Created with ❤️ by [Llumo](https://www.llumo.ai/)')

# Function to load and extract text from a PDF
def load_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text=text)

# Function to load or create embeddings
def load_embeddings(store_name, chunks):
    embeddings = OpenAIEmbeddings()
    index_path = f"{store_name}.index"
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(index_path)
    return vector_store

# Function to compress text using Llumo API
def compress_with_llumo(text, topic=None):
    LLUMO_API_KEY = os.getenv('LLUMO_API_KEY')
    LLUMO_ENDPOINT = "https://app.llumo.ai/api/compress"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLUMO_API_KEY}"
    }
    payload = {"prompt": text}
    
    if topic:
        payload["topic"] = topic
    
    try:
        response = requests.post(LLUMO_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        
        # Check for 'data' key in the response
        if 'data' in result and 'data' in result['data']:
            data = result['data']['data']
            data_content = json.loads(data)
            
            compressed_text = data_content.get('compressedPrompt', text)
            initial_tokens = data_content.get('initialTokens', 0)
            final_tokens = data_content.get('finalTokens', 0)
            
            if initial_tokens and final_tokens:
                compression_percentage = ((initial_tokens - final_tokens) / initial_tokens) * 100
            else:
                compression_percentage = 0
            
            return compressed_text, True, compression_percentage, initial_tokens, final_tokens
        else:
            raise KeyError("'data' key not found in the response")
    
    except (json.JSONDecodeError, requests.RequestException, KeyError) as e:
        logger.error(f"Error compressing text: {str(e)}")
        st.error(f"Error compressing text: {str(e)}")
        return text, False, 0, 0, 0

# Main function
def main():
    st.header("Chat with Your PDF 📄")
    setup_sidebar()
    load_dotenv()
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # File uploader
    pdf = st.file_uploader("Upload your PDF file here", type='pdf')
   
    if pdf is not None:
        # Process the uploaded PDF
        text = load_pdf(pdf)
        chunks = split_text(text)
        st.write(f"Number of chunks: {len(chunks)}")
       
        store_name = pdf.name[:-4]
        st.write(f'Processing file: **{store_name}**')
       
        # Load or create embeddings
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = load_embeddings(store_name, chunks)
       
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # User query input
        query = st.chat_input("Ask any question related to the content of the PDF:")
        if query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            
            # Perform similarity search
            docs = st.session_state.vector_store.similarity_search(query=query, k=3)
           
            # Combine retrieved documents into a single context
            context = " ".join([doc.page_content for doc in docs])
            
            # Compress the context using Llumo
            compressed_text, success, compression_percentage, initial_tokens, final_tokens = compress_with_llumo(context, topic=query)
            
            if success:
                # Use ChatOpenAI for question answering
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                
                # Get response and display answer
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=[Document(page_content=compressed_text)], question=query)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.write(response)
                    
                    st.write(f"Cost of this query: ${cb.total_cost:.5f}")
                    st.write(f"Compression achieved: {compression_percentage:.2f}%")
                    st.write(f"Initial tokens: {initial_tokens}")
                    st.write(f"Final tokens: {final_tokens}")
            else:
                # Use original context if compression fails
                st.error("Failed to compress the text. Using original context.")
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.write(response)
                    
                    st.write(f"Cost of this query: ${cb.total_cost:.5f}")

# Run the main function
if __name__ == '__main__':
    main()