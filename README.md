# PDF Chat Assistant and Llumo Integration

## Overview

Welcome to the PDF Chat Assistant and Llumo Integration repository. This project combines natural language processing, text compression, and vector similarity search to enable interactive querying of PDF documents. It leverages OpenAI's GPT models and Llumo's text compression API to deliver efficient and effective results.

## Projects

### 1. PDF Chat Assistant

A Streamlit-based web application that allows users to chat with their PDF documents. Key features include:
- **PDF Upload**: Easily upload PDF files.
- **Text Extraction**: Extract text content from PDF pages.
- **Text Chunking**: Split text into manageable chunks.
- **Embedding**: Convert text chunks into vector embeddings.
- **Vector Storage**: Efficiently search for similar text using FAISS.
- **Question Answering**: Utilize OpenAI's GPT models for answering questions based on the PDF content.
- **Text Compression**: Use Llumo's API to compress text and optimize context for queries.

### 2. Llumo Integration

An integration that provides text compression services via Llumo's API. Key features include:
- **API Interaction**: Connect with Llumo's compression endpoint.
- **Context Optimization**: Compress text to reduce token usage.
- **Compression Statistics**: Display detailed compression metrics.

## Setup and Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Steps
1. **Clone the Repository**:
    ```shell
    git clone <repository-url>
    cd <project-directory>
    ```
2. **Set Up a Virtual Environment**:
    ```shell
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```
3. **Install Dependencies**:
    ```shell
    pip install -r requirements.txt
    ```
4. **Configure Environment Variables**:
    Create a `.env` file in the project root:
    ```shell
    OPENAI_API_KEY=your_openai_api_key_here
    LLUMO_API_KEY=your_llumo_api_key_here
    ```

### Dependencies
- `langchain`
- `langchain-openai`
- `langchain-community`
- `PyPDF2`
- `python-dotenv`
- `streamlit`
- `faiss-cpu`
- `openai`
- `streamlit-extras`

## Running the Application
1. **Start the Streamlit Server**:
    ```shell
    streamlit run app.py
    ```
2. **Access the Web Interface**:
    Open your web browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature.
3. Implement your feature or bug fix.
4. Add or update tests as necessary.
5. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Llumo for providing the text compression API.
- OpenAI for their powerful language models.
- The LangChain community for the extensive framework.
