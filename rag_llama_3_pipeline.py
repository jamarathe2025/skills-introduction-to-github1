import os
import requests
import json
from pypdf import PdfReader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Custom embedding class using Ollama REST API
class OllamaEmbeddings(Embeddings):
    def __init__(self, model='llama-3.2', url='http://localhost:11434/api/embeddings'):
        self.model = model
        self.url = url

    def embed(self, text):
        headers = {'Content-Type': 'application/json'}
        payload = {'model': self.model, 'input': text}
        response = requests.post(self.url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_json = response.json()
            if 'embedding' in response_json:
                return response_json['embedding']
            else:
                raise ValueError("'embedding' not found in response")
        else:
            raise Exception(f"Error embedding text: {response.status_code} - {response.text}")

    def embed_documents(self, texts):
        return [self.embed(text) for text in texts]

    def embed_query(self, text):
        return self.embed(text)

# Load text from a PDF file
def load_pdf(file_path, num_pages=None):
    reader = PdfReader(file_path)
    text = ""
    pages = reader.pages[:num_pages] if num_pages else reader.pages
    for page in pages:
        text += page.extract_text() + "\n"
    return text

# Define LLM
llm = Ollama(model_name='llama-3.2', temperature=0)

# Path to your PDF
document_text = load_pdf('path/to/your/pdf/document.pdf', num_pages=5)

# Split text into chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=256, chunk_overlap=32)
text_chunks = text_splitter.split_text(document_text)

# Convert to LangChain documents
documents = [Document(page_content=chunk) for chunk in text_chunks]

# Embed documents
embeddings = OllamaEmbeddings(model='llama-3.2', url='http://localhost:11434/api/embeddings')

# Create FAISS index
knowledge_base = FAISS.from_documents(documents, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=knowledge_base.as_retriever(),
    return_source_documents=True
)

# Ask a question
question = "What is the main topic of the document?"
response = qa_chain.invoke({"query": question})
print("Response:", response['result'])

# Display response (for Jupyter Notebook)
from IPython.display import display, HTML
display(HTML("<h2>Response</h2>"))
display(HTML(f"<p>{response['result']}</p>"))
