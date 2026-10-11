import os
import argparse
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGretriver
from src.pipeline import RAGpipeline




load_dotenv()

def run_sync(data_dir: str, processor: DocumentProcessor, embedder: EmbeddingManager, vector_store: VectorStore):
    """Detects, processes, and embeds only new or modified files."""
    new_docs, processed_paths = processor.process_all_files(data_directory=data_dir)

    if not new_docs:
        print("Everything is up to date. No new files to ingest.")
        return

    chunks = processor.split_documents(new_docs)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)
    print(f"Sync complete. New files indexed: {len(processed_paths)}.")

def main():
    parser = argparse.ArgumentParser(description="Multi-format RAG Application")
    parser.add_argument("--sync", action="store_true", help="Sync/index new or modified files from data folder")
    args = parser.parse_args()

    api_key = os.getenv("groq_api")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in .env file.")

    llm = ChatGroq(
        groq_api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1024
    )

    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200, tracker_file="data/processed_files.json")
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    store = VectorStore(collection_name="multi_format_docs", db_directory="data/VectorStore")
    retriever = RAGretriver(vector_store=store, embedding_manager=embedder)
    pipeline = RAGpipeline(rag_retriver=retriever, llm=llm)

    # Automatically check for updates on startup or when explicit flag is passed
    if args.sync or store.count() == 0:
        print("\nChecking for new files in data/...")
        run_sync("data", processor, embedder, store)

    print(f"\nRAG system ready! ({store.count()} chunks in store)")
    while True:
        try:
            query = input("\nQuestion (or 'exit' to quit): ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break

            result = pipeline.query(query, top_k=3, min_score=0.1, stream=True, summarize=False)
            if not result["sources"]:
                print(result["answer"])

        except KeyboardInterrupt:
            print("\nShutting down.")
            break

if __name__ == "__main__":
    main()