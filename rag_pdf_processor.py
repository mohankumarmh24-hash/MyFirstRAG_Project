# rag_pdf_processor.py

import os
import PyPDF2
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from typing import List, Dict, Any
import tempfile

class PDFRAGSystem:
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.vector_store = None
        self.qa_chain = None
        self.documents = []
        self.setup_components()
    
    def setup_components(self):
        """Initialize all necessary components"""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM
        if self.groq_api_key:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.1
            )
        else:
            raise ValueError("GROQ API key is required. Please provide it in constructor or set GROQ_API_KEY environment variable.")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_pdf(self, pdf_path: str) -> List[str]:
        """Load and extract text from PDF file"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            self.documents.extend(documents)
            print(f"✅ Loaded {len(documents)} pages from {pdf_path}")
            return documents
        except Exception as e:
            print(f"❌ Error loading PDF: {str(e)}")
            # Fallback to PyPDF2
            return self._load_with_pypdf2(pdf_path)
    
    def _load_with_pypdf2(self, pdf_path: str) -> List[str]:
        """Fallback PDF loading using PyPDF2"""
        try:
            documents = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        documents.append(text)
            print(f"✅ Loaded {len(documents)} pages using PyPDF2")
            return documents
        except Exception as e:
            print(f"❌ Error with PyPDF2: {str(e)}")
            return []
    
    def process_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        # Convert to text if they're document objects
        if hasattr(documents[0], 'page_content'):
            texts = [doc.page_content for doc in documents]
        else:
            texts = documents
        
        chunks = self.text_splitter.split_text("\n".join(texts))
        print(f"✅ Split into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List[str]):
        """Create FAISS vector store from text chunks"""
        if not chunks:
            raise ValueError("No text chunks available to create vector store")
        
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        print("✅ Vector store created successfully")
        return self.vector_store
    
    def setup_qa_chain(self, custom_prompt: str = None):
        """Setup the QA chain with custom prompt"""
        
        # Default prompt template
        default_prompt = """
        You are an AI assistant that answers questions based on the provided context from PDF documents.

        Context: {context}

        Question: {question}

        Instructions:
        1. Answer the question based ONLY on the provided context
        2. If the context doesn't contain relevant information, say "I cannot find this information in the provided documents"
        3. Provide clear, concise answers
        4. Include relevant details and examples from the context when available
        5. If referring to specific information, mention which part of the document it came from

        Answer:
        """
        
        prompt_template = custom_prompt or default_prompt
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("✅ QA chain setup completed")
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please call setup_qa_chain() first.")
        
        result = self.qa_chain({"query": question})
        return result
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of processed documents"""
        if not self.documents:
            return {"error": "No documents processed"}
        
        total_pages = len(self.documents)
        total_chunks = len(self.text_splitter.split_text(
            "\n".join([doc.page_content if hasattr(doc, 'page_content') else doc for doc in self.documents])
        )) if self.documents else 0
        
        return {
            "total_documents": 1,  # For single PDF
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "vector_store_ready": self.vector_store is not None,
            "qa_chain_ready": self.qa_chain is not None
        }

# Streamlit Web Interface
class PDFRAGInterface:
    def __init__(self):
        self.rag_system = None
        self.setup_page()
    
    def setup_page(self):
        st.set_page_config(
            page_title="PDF RAG Query System",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("📚 PDF RAG Query System")
        st.markdown("Upload a PDF and ask questions about its content using AI")
    
    def initialize_system(self, groq_api_key: str):
        """Initialize the RAG system"""
        try:
            self.rag_system = PDFRAGSystem(groq_api_key)
            return True
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            return False
    
    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.title("⚙️ Configuration")
        
        groq_api_key = st.sidebar.text_input(
            "GROQ API Key",
            type="password",
            help="Get your free API key from https://groq.com"
        )
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload PDF File",
            type=['pdf'],
            help="Upload the PDF you want to query"
        )
        
        process_btn = st.sidebar.button("Process PDF")
        
        return groq_api_key, uploaded_file, process_btn
    
    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded PDF file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load and process PDF
            documents = self.rag_system.load_pdf(tmp_path)
            chunks = self.rag_system.process_documents(documents)
            self.rag_system.create_vector_store(chunks)
            self.rag_system.setup_qa_chain()
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return True
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return False
    
    def display_document_info(self):
        """Display information about the processed document"""
        if self.rag_system and self.rag_system.documents:
            summary = self.rag_system.get_document_summary()
            
            st.sidebar.success("✅ PDF Processed Successfully!")
            st.sidebar.subheader("Document Info")
            st.sidebar.write(f"📄 Pages: {summary['total_pages']}")
            st.sidebar.write(f"🔢 Chunks: {summary['total_chunks']}")
            st.sidebar.write(f"🔍 Vector Store: {'✅ Ready' if summary['vector_store_ready'] else '❌ Not Ready'}")
            st.sidebar.write(f"🤖 QA System: {'✅ Ready' if summary['qa_chain_ready'] else '❌ Not Ready'}")
    
    def chat_interface(self):
        """Main chat interface"""
        st.header("💬 Ask Questions About Your PDF")
        
        if not self.rag_system or not self.rag_system.qa_chain:
            st.info("👈 Please upload and process a PDF file first using the sidebar.")
            return
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.write(f"**Source {i+1}:**")
                            st.text(source[:500] + "..." if len(source) > 500 else source)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Searching through document..."):
                    try:
                        result = self.rag_system.query(prompt)
                        answer = result['result']
                        
                        # Extract source documents
                        sources = [doc.page_content for doc in result['source_documents']]
                        
                        st.markdown(answer)
                        
                        # Add to chat history with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    
                    except Exception as e:
                        error_msg = f"Error processing your question: {str(e)}"
                        st.error(error_msg)
    
    def example_queries(self):
        """Show example queries"""
        st.sidebar.subheader("💡 Example Queries")
        
        examples = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the main findings or conclusions?",
            "List the important recommendations",
            "Explain the methodology used"
        ]
        
        for example in examples:
            if st.sidebar.button(example, key=example):
                st.session_state.auto_query = example
    
    def run(self):
        """Main application runner"""
        groq_api_key, uploaded_file, process_btn = self.sidebar_controls()
        
        # Handle PDF processing
        if process_btn and uploaded_file and groq_api_key:
            with st.spinner("Processing PDF..."):
                if self.initialize_system(groq_api_key):
                    if self.process_uploaded_file(uploaded_file):
                        self.display_document_info()
        
        # Show example queries
        self.example_queries()
        
        # Main chat interface
        self.chat_interface()

# Advanced RAG with Multiple PDFs
class MultiPDFRAGSystem(PDFRAGSystem):
    def __init__(self, groq_api_key: str):
        super().__init__(groq_api_key)
        self.processed_files = []
    
    def load_multiple_pdfs(self, pdf_directory: str):
        """Load multiple PDFs from a directory"""
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        all_documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            documents = self.load_pdf(pdf_path)
            all_documents.extend(documents)
            self.processed_files.append(pdf_file)
        
        return all_documents
    
    def query_with_source_awareness(self, question: str):
        """Enhanced query that includes source document information"""
        result = self.query(question)
        
        # Add source file information
        source_files = set()
        for doc in result['source_documents']:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_files.add(os.path.basename(doc.metadata['source']))
        
        result['source_files'] = list(source_files)
        return result

# Usage Examples and Demo
def demo_single_pdf():
    """Demo with a single PDF"""
    rag_system = PDFRAGSystem(groq_api_key="your-groq-api-key")
    
    # Process PDF
    documents = rag_system.load_pdf("sample_patient.pdf")
    chunks = rag_system.process_documents(documents)
    rag_system.create_vector_store(chunks)
    rag_system.setup_qa_chain()
    
    # Query examples
    queries = [
        "What is the patient's main medical condition?",
        "What medications is the patient taking?",
        "Summarize the patient's medical history",
        "What are the recent test results?"
    ]
    
    print("🔍 PDF RAG System Demo")
    print("=" * 50)
    
    for query in queries:
        print(f"\n❓ Question: {query}")
        result = rag_system.query(query)
        print(f"🤖 Answer: {result['result']}")
        print(f"📚 Sources: {len(result['source_documents'])} relevant chunks")
        print("-" * 80)

def demo_custom_prompt():
    """Demo with custom prompt template"""
    
    custom_prompt = """
    You are a medical assistant analyzing patient records. Use the context below to answer questions.

    MEDICAL CONTEXT:
    {context}

    QUESTION: {question}

    Please provide:
    1. A clear medical assessment based on the context
    2. Relevant patient information (age, conditions, medications)
    3. Any concerning findings that need attention
    4. Recommendations for follow-up if mentioned

    If information is missing, state what additional details would be helpful.

    MEDICAL ASSESSMENT:
    """
    
    rag_system = PDFRAGSystem(groq_api_key="your-groq-api-key")
    documents = rag_system.load_pdf("sample_patient.pdf")
    chunks = rag_system.process_documents(documents)
    rag_system.create_vector_store(chunks)
    rag_system.setup_qa_chain(custom_prompt)
    
    result = rag_system.query("What is the patient's current health status?")
    print("Custom Prompt Result:")
    print(result['result'])

if __name__ == "__main__":
    # Run Streamlit app
    app = PDFRAGInterface()
    app.run()
    
    # Uncomment for direct demos
    # demo_single_pdf()
    # demo_custom_prompt()
