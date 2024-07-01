# PDF Chat Assistant

## Project Overview

The PDF Chat Assistant is a Streamlit-based web application that allows you to chat with your PDF documents. By using natural language processing and machine learning technologies, you can upload PDFs, ask questions, and get answers based on the content.

## Features

### 1. PDF Processing
- **Upload**: Upload PDF files easily.
- **Text Extraction**: Extract text content from each page of the PDF using PyPDF2.

### 2. Text Processing
- **Chunking**: Break down large texts into manageable chunks with RecursiveCharacterTextSplitter from LangChain.
- **Embedding**: Convert text chunks into vector representations using OpenAIEmbeddings.

### 3. Vector Storage
- **FAISS Integration**: Use FAISS (Facebook AI Similarity Search) for efficient similarity search operations.
- **Persistent Storage**: Save and load vector stores from disk for faster subsequent runs.

### 4. Llumo Text Compression
- **API Integration**: Connect with Llumo's API for text compression.
- **Context Optimization**: Compress retrieved context to reduce token usage.

### 5. Question Answering
- **OpenAI Integration**: Utilize OpenAI's GPT models through the ChatOpenAI interface.
- **Chain Creation**: Implement a question-answering chain using LangChain's load_qa_chain.

### 6. User Interface
- **Streamlit Framework**: Provide an intuitive, interactive web interface.
- **Real-time Feedback**: Display compression statistics, costs, and processing steps.

## Component Breakdown

### PDF Processing
The `load_pdf` function uses PyPDF2's PdfReader to extract text content from each page of the uploaded PDF.

### Text Chunking
The `split_text` function employs LangChain's RecursiveCharacterTextSplitter to break down the extracted text into smaller chunks (1000 characters each with a 200-character overlap).

### Embedding and Vector Storage
The `load_embeddings` function creates or loads a vector store:
1. Checks if a saved index exists for the given PDF.
2. Loads pre-computed embeddings using FAISS if available.
3. Creates new embeddings using OpenAIEmbeddings if not available.
4. Saves the resulting vector store locally for future use.

### Llumo Integration
The `compress_with_llumo` function handles the interaction with Llumo's API:
1. Prepares and sends a POST request to Llumo's compression endpoint.
2. Parses the JSON response and extracts compressed text and token counts.
3. Calculates the compression percentage and returns the compressed text along with statistics.

### Question Answering Process
1. Retrieves the most relevant text chunks based on the user's query.
2. Combines these chunks into a single context.
3. Compresses the context using Llumo.
4. Uses ChatOpenAI with the GPT-3.5-turbo model to generate an answer.
5. Tracks and displays the cost of the OpenAI API call.

## Setup and Installation

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
- `langchain` (v0.1.7)
- `langchain-openai` (v0.0.5)
- `langchain-community` (v0.0.20)
- `PyPDF2` (v3.0.1)
- `python-dotenv` (v1.0.1)
- `streamlit` (v1.29.0)
- `faiss-cpu` (v1.7.4)
- `openai` (v1.12.0)
- `streamlit-extras` (v0.3.6)

## Running the Application
1. **Start the Streamlit Server**:
    ```shell
    streamlit run app.py
    ```
2. **Access the Web Interface**:
    Open your web browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## Usage Guide
1. **PDF Upload**:
    - Click the "Upload your PDF file here" button.
    - Select a PDF file from your local system.
2. **Processing Feedback**:
    - The app displays the number of text chunks created and the name of the file being processed.
3. **Asking Questions**:
    - Type your question into the chat input field.
    - Press Enter to submit your question.
4. **Viewing Results**:
    - The app displays the original and compressed context, compression statistics, and the answer to your question.
5. **Iterative Questioning**:
    - You can ask multiple questions about the same PDF without re-uploading.

## Troubleshooting

### Common Issues and Solutions
1. **API Key Errors**:
    - Ensure your `.env` file contains valid API keys.
2. **Dependency Conflicts**:
    - Create a fresh virtual environment and reinstall dependencies if needed.
3. **PDF Processing Errors**:
    - Ensure your PDF is not encrypted or password-protected.
4. **Out of Memory Errors**:
    - Adjust chunking parameters for large PDFs.

## Performance Optimization
- Saves the vector store locally after the first run for faster subsequent uses.
- Llumo's text compression helps reduce token usage, potentially lowering costs.

## Security Considerations
- Store API keys in the `.env` file, which should not be committed to version control.
- Ensure the Streamlit app is not publicly accessible if handling sensitive documents.

## Future Enhancements
1. Multi-PDF support.
2. Integration with other language models.
3. Advanced PDF handling (e.g., table extraction, image analysis).
4. User authentication for enhanced security.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature.
3. Implement your feature or bug fix.
4. Add or update tests as necessary.
5. Submit a pull request with a clear description of your changes.

## Acknowledgments
- Llumo for providing the text compression API.
- OpenAI for their powerful language models.
- The LangChain community for the extensive framework.
