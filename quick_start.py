# quick_start.py

from rag_pdf_processor import PDFRAGSystem
import os

def quick_start(pdf_path: str, groq_api_key: str):
    """
    Quick start function to set up RAG system in one line
    """
    # Initialize system
    rag = PDFRAGSystem(groq_api_key)
    
    # Process PDF
    print("📥 Loading PDF...")
    documents = rag.load_pdf(pdf_path)
    
    print("🔪 Splitting text...")
    chunks = rag.process_documents(documents)
    
    print("🔍 Creating vector store...")
    rag.create_vector_store(chunks)
    
    print("🤖 Setting up QA system...")
    rag.setup_qa_chain()
    
    print("✅ RAG system ready!")
    return rag

# Example usage
if __name__ == "__main__":
    # Replace with your actual API key and PDF path
    API_KEY = "your-groq-api-key"
    PDF_PATH = "sample_patient.pdf"
    
    # One-line setup
    rag_system = quick_start(PDF_PATH, API_KEY)
    
    # Ask questions
    while True:
        question = input("\n❓ Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = rag_system.query(question)
        print(f"\n🤖 Answer: {answer['result']}")
