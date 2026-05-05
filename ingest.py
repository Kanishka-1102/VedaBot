from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data/'

def create_vector_db():
    """Create vector database from PDF documents with improved chunking strategy."""
    print("🔄 Loading PDF documents...")
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)

    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")

    # Improved chunking: larger chunks with more overlap for better context
    # Chunk size of 1000 characters captures more context while staying within token limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Increased from 500 for better context
        chunk_overlap=100,     # Increased from 50 for better continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting strategy
    )

    print("🔄 Splitting documents into chunks...")
    texts = text_splitter.split_documents(documents)
    print(f"✅ Created {len(texts)} text chunks")

    print("🔄 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    DB_FAISS_PATH = 'vectorstore/db_faiss'

    print("🔄 Building vector database...")
    db = FAISS.from_documents(texts, embeddings)

    print(f"🔄 Saving vector database to {DB_FAISS_PATH}...")
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vector database created successfully with {db.index.ntotal} vectors!")

if __name__ == "__main__":
    create_vector_db()
