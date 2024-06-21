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
        st.write('Created with ❤️ by [Llumo](https://www.llumo.ai/)')


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
    embeddings = OpenAIEmbeddings()
    index_path = f"{store_name}.index"
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(index_path)
    return vector_store


def main():
    st.header("Interact with Your PDF 📄")
    setup_sidebar()
    load_dotenv()


    pdf = st.file_uploader("Upload your PDF file here", type='pdf')
   
    if pdf is not None:
        text = load_pdf(pdf)
        # for debugging purpose
        # st.write("PDF Content Preview (first 500 characters):")
        # st.write(text[:500] + "...")
       
        chunks = split_text(text)
        st.write(f"Number of chunks: {len(chunks)}")
       
        store_name = pdf.name[:-4]
        st.write(f'Processing file: **{store_name}**')
       
        # Use st.session_state to store the vector_store
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = load_embeddings(store_name, chunks)
       
        query = st.text_input("Ask any question related to the content of the PDF:")
        if query:
            docs = st.session_state.vector_store.similarity_search(query=query, k=3)
           
            # for debugging
            # st.write("### Retrieved Contexts:")
            # for i, doc in enumerate(docs):
            #     st.write(f"Context {i+1}:")
            #     st.write(doc.page_content)
           
            # Use ChatOpenAI instead of OpenAI
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
           
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write("### Answer")
                st.write(response)
                st.write(f"Cost of this query: ${cb.total_cost:.5f}")


if __name__ == '__main__':
    main()
