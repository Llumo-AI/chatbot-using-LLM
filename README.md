# PDF Query Assistant

## Project Overview

The PDF Query Assistant is an advanced Streamlit-based web application designed to revolutionize how users interact with PDF documents. By leveraging cutting-edge natural language processing and machine learning technologies, this tool allows users to upload PDFs, ask questions about their content, and receive accurate, context-aware answers.

## Detailed Features

### 1. PDF Processing
- **Upload**: Users can upload PDF files through a simple interface.
- **Text Extraction**: Utilizes PyPDF2 to extract text content from each page of the uploaded PDF.

### 2. Text Processing
- **Chunking**: Implements RecursiveCharacterTextSplitter from LangChain to break down large texts into manageable chunks.
- **Embedding**: Uses OpenAIEmbeddings to convert text chunks into vector representations.

### 3. Vector Storage
- **FAISS Integration**: Employs FAISS (Facebook AI Similarity Search) for efficient similarity search operations.
- **Persistent Storage**: Saves and loads vector stores to/from disk for faster subsequent runs.

### 4. Llumo Text Compression
- **API Integration**: Connects with Llumo's API for advanced text compression.
- **Context Optimization**: Compresses retrieved context to reduce token usage and potentially lower costs.

### 5. Question Answering
- **OpenAI Integration**: Utilizes OpenAI's GPT models through the ChatOpenAI interface.
- **Chain Creation**: Implements a question-answering chain using LangChain's load_qa_chain.

### 6. User Interface
- **Streamlit Framework**: Provides an intuitive, interactive web interface.
- **Real-time Feedback**: Displays compression statistics, costs, and processing steps.

## Detailed Component Breakdown

### PDF Processing
The `load_pdf` function uses PyPDF2's PdfReader to iterate through each page of the uploaded PDF, extracting text content. This raw text serves as the basis for all subsequent processing.

### Text Chunking
The `split_text` function employs LangChain's RecursiveCharacterTextSplitter. It breaks down the extracted text into smaller, overlapping chunks (1000 characters each with a 200-character overlap). This chunking is crucial for managing large documents and enabling more precise information retrieval.

### Embedding and Vector Storage
The `load_embeddings` function is responsible for creating or loading a vector store:
1. It first checks if a saved index exists for the given PDF.
2. If it exists, it loads the pre-computed embeddings using FAISS.
3. If not, it creates new embeddings using OpenAIEmbeddings and stores them using FAISS.
4. The resulting vector store is saved locally for future use.

### Llumo Integration

The `compress_with_llumo` function handles the interaction with Llumo's API:

1. **API Call Preparation**:
   - Constructs the API endpoint URL and headers, including the Llumo API key.
   - Prepares the payload with the text to be compressed and optionally includes the topic (user's query).

2. **API Interaction**:
   - Sends a POST request to Llumo's compression endpoint.
   - Handles potential errors (network issues, API errors, unexpected response formats).

3. **Response Processing**:
   - Parses the JSON response from Llumo.
   - Extracts the compressed text, along with token counts before and after compression.

4. **Compression Statistics**:
   - Calculates the compression percentage.
   - Returns the compressed text along with compression statistics.

### Question Answering Process

1. **Context Retrieval**:
   - Uses the vector store to find the most relevant text chunks based on the user's query.
   - Combines these chunks into a single context.

2. **Context Compression**:
   - Sends the retrieved context to Llumo for compression.
   - If compression is successful, uses the compressed text for the next steps.

3. **Language Model Integration**:
   - Initializes ChatOpenAI with the GPT-3.5-turbo model.
   - Creates a question-answering chain using LangChain.

4. **Answer Generation**:
   - Runs the QA chain with the (compressed) context and user's question.
   - Tracks and displays the cost of the OpenAI API call.

## Setup and Installation

### Detailed Steps

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

### Dependency Details

- `langchain` (v0.1.7): Provides the framework for creating the question-answering pipeline.
- `langchain-openai` (v0.0.5): Offers OpenAI-specific integrations for LangChain.
- `langchain-community` (v0.0.20): Includes community-contributed components for LangChain.
- `PyPDF2` (v3.0.1): Used for PDF text extraction.
- `python-dotenv` (v1.0.1): Loads environment variables from the .env file.
- `streamlit` (v1.29.0): Powers the web interface.
- `faiss-cpu` (v1.7.4): Enables efficient similarity search for embeddings.
- `openai` (v1.12.0): Provides access to OpenAI's API.
- `streamlit-extras` (v0.3.6): Offers additional Streamlit components.

## Running the Application

1. **Start the Streamlit Server**:
```shell
streamlit run app.py
```
2. **Access the Web Interface**:
Open your web browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## Usage Guide

1. **PDF Upload**:
- Click the "Upload your PDF file here" button in the main area.
- Select a PDF file from your local system.

2. **Processing Feedback**:
- The app will display the number of text chunks created and the name of the file being processed.

3. **Asking Questions**:
- Type your question into the text input field labeled "Ask any question related to the content of the PDF:".
- Press Enter or click outside the input field to submit your question.

4. **Viewing Results**:
- The app will display:
  - Original and compressed context
  - Compression statistics (percentage, token counts)
  - The answer to your question
  - The cost of the OpenAI API call for this query

5. **Iterative Questioning**:
- You can ask multiple questions about the same PDF without reuploading.

## Troubleshooting

### Common Issues and Solutions

1. **API Key Errors**:
- Ensure your `.env` file is in the correct location and contains valid API keys.
- Check for any spaces or quotes around your API keys.

2. **Dependency Conflicts**:
- If you encounter module import errors, try creating a fresh virtual environment and reinstalling dependencies.

3. **PDF Processing Errors**:
- Ensure your PDF is not encrypted or password-protected.
- For scanned PDFs, OCR might be necessary (not included in this tool).

4. **Out of Memory Errors**:
- For very large PDFs, you might need to increase your system's RAM or adjust the chunking parameters.

## Performance Optimization

- The vector store is saved locally after the first run, significantly speeding up subsequent uses with the same PDF.
- Llumo's text compression helps reduce the number of tokens processed by the OpenAI model, potentially lowering costs for large documents.

## Security Considerations

- API keys are stored in the `.env` file, which should never be committed to version control.
- Ensure your Streamlit app is not publicly accessible if you're dealing with sensitive documents.

## Future Enhancements

Potential areas for improvement:
1. Multi-PDF support for querying across multiple documents.
2. Integration with other language models (e.g., Hugging Face models).
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