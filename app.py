import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
def setup_sidebar():
    with st.sidebar:
        st.title('📚💬 PDF Query Assistant')
        st.markdown('''
        ## About
        This app allows you to interact with your PDF documents using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) language models
        
        Upload a PDF and ask questions to get answers instantly!
        ''')
        add_vertical_space(5)
        st.write('Created with ❤️ by [Llumo](https://www.llumo.ai//)')

def load_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text=text)

def load_embeddings(store_name, chunks):
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
    return vector_store

def main():
    st.header("Interact with Your PDF 📄")
    setup_sidebar()
    load_dotenv()

    pdf = st.file_uploader("Upload your PDF file here", type='pdf')
    if pdf is not None:
        text = load_pdf(pdf)
        chunks = split_text(text)
        store_name = pdf.name[:-4]
        st.write(f'Processing file: **{store_name}**')
        vector_store = load_embeddings(store_name, chunks)
        
        query = st.text_input("Ask any question related to the content of the PDF:")
        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write("### Answer")
            st.write(response)

if __name__ == '__main__':
    main()
