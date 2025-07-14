# rag_pipeline.py

import faiss
from uuid import uuid4

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


### 📄 Component 1: Document Loader
class PdfLoader:
    def read_file(self, filepath):
        loader = PyMuPDFLoader(filepath)
        return loader.load()


### ✂️ Component 2: Text Chunker
class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_docs(self, docs):
        chunks = []
        for doc in docs:
            split_texts = self.text_splitter.split_text(doc.page_content)
            for chunk in split_texts:
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        return chunks


### 🔍 Component 3: Embedding Model
class Embeddings:
    def __init__(self, model_name="llama3.2"):
        self.embeddings = OllamaEmbeddings(model=model_name)

    def embed(self, texts):
        return self.embeddings.embed_documents(texts)

    def get_embed_model(self):
        return self.embeddings


### 🧠 Component 4: Vector Store
class VectorStore:
    def __init__(self, embedding_model):
        sample_dim = len(embedding_model.embed_query("sample"))
        self.index = faiss.IndexFlatL2(sample_dim)
        self.vector_store = FAISS(
            embedding_function=embedding_model,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_documents(self, documents):
        ids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=ids)

    def search(self, query, k=5):
        return self.vector_store.similarity_search(query, k=k)


### 🔁 Component 5: Retriever
class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, k=5):
        return self.vector_store.search(query, k)


### 🧩 Component 6: Prompt + LLM + Pipeline
class RAG:
    def __init__(self, model_name="llama3.2"):
        self.prompt_template = """Instruction: You're an expert problem solver. You only use the context provided to answer. Be honest — if the answer isn't in the context, say "I don't know."
User Question: {user_query}
Answer Context: {answer_context}
"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.llm = OllamaLLM(model=model_name)

        self.loader = PdfLoader()
        self.chunker = Chunker()
        self.embeddings = Embeddings(model_name)
        self.vector_store = VectorStore(self.embeddings.get_embed_model())
        self.retriever = Retriever(self.vector_store)

    def run(self, filepath, query, k=5):
        # Load and chunk
        print("🔄 Loading PDF...")
        docs = self.loader.read_file(filepath)
        print(f"✅ Loaded {len(docs)} pages.")

        print("✂️ Chunking text...")
        chunks = self.chunker.chunk_docs(docs)
        print(f"✅ Created {len(chunks)} chunks.")

        print("🧠 Embedding and indexing...")
        self.vector_store.add_documents(chunks)
        print("✅ Documents added to vector store.")

        print(f"📡 Retrieving top {k} relevant chunks...")
        retrieved_docs = self.retriever.retrieve(query, k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        print("💬 Generating response...")
        chain = self.prompt | self.llm
        response = chain.invoke({
            "user_query": query,
            "answer_context": context,
        })

        return response


### 🏁 Entry Point
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run RAG pipeline on a PDF")
    parser.add_argument("--model", type=str, help="LLM model name for Ollama")
    parser.add_argument("--file", type=str, help="Path to PDF file")
    parser.add_argument("--query", type=str, help="User query")
    parser.add_argument("--k", type=int, help="Number of documents to retrieve")

    args = parser.parse_args()

    # If arguments not provided, prompt user
    if not all([args.model, args.file, args.query, args.k]):
        print("⚙️ No command-line arguments detected. Please enter them now.")
        args.model = args.model or input("🤖 Model name (default: llama3.2): ") or "llama3.2"
        args.file = args.file or input("📄 Path to PDF file: ")
        args.query = args.query or input("❓ User query: ")
        args.k = args.k or int(input("🔢 Number of documents to retrieve (k): "))

    # Confirm choices
    print("\n🚀 Running with:")
    print(f"  🔧 Model: {args.model}")
    print(f"  📄 File: {args.file}")
    print(f"  ❓ Query: {args.query}")
    print(f"  🔢 Top K: {args.k}\n")

    # Run RAG pipeline
    rag = RAG(model_name=args.model)
    answer = rag.run(filepath=args.file, query=args.query, k=args.k)
    
    print("\n🧠 FINAL RESPONSE:\n")
    print(answer)


