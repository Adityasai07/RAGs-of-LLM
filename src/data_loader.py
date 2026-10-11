import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tracker_file: str = "data/processed_files.json"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tracker_file = tracker_file
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.processed_records: Dict[str, str] = self._load_tracker()

    def _load_tracker(self) -> Dict[str, str]:
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_tracker(self):
        os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
        with open(self.tracker_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_records, f, indent=4)

    @staticmethod
    def _get_file_hash(file_path: Path) -> str:
        """Calculate MD5 hash to detect file changes."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_single_file(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()
        docs = []

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

            elif ext in [".txt", ".sql", ".md", ".json", ".py"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

            elif ext == ".csv":
                df = pd.read_csv(file_path)
                # Convert rows or structured summary to document text
                content = df.to_string(index=False)
                docs = [Document(page_content=content)]

            elif ext in [".xlsx", ".xls"]:
                excel_sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in excel_sheets.items():
                    content = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
                    docs.append(Document(page_content=content, metadata={"sheet_name": sheet_name}))

            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["file_type"] = ext.replace(".", "")
                doc.metadata["source_path"] = str(file_path)

            return docs

        except Exception as e:
            print(f"  ✗ Error loading {file_path.name}: {e}")
            return []

    def process_all_files(self, data_directory: str) -> Tuple[List[Document], List[Path]]:
        """Scans folder, filters for new/updated files, loads, and tracks them."""
        all_documents: List[Document] = []
        newly_processed_files: List[Path] = []
        data_dir = Path(data_directory)

        supported_extensions = {".pdf", ".txt", ".sql", ".csv", ".xlsx", ".xls", ".md", ".json"}
        all_files = [f for f in data_dir.glob("**/*") if f.is_file() and f.suffix.lower() in supported_extensions]

        print(f"Found {len(all_files)} total supported files in {data_directory}")

        for file_path in all_files:
            file_key = str(file_path.resolve())
            current_hash = self._get_file_hash(file_path)

            # Check if file has already been ingested without modification
            if file_key in self.processed_records and self.processed_records[file_key] == current_hash:
                continue

            print(f"Processing new or modified file: {file_path.name}")
            docs = self._load_single_file(file_path)
            if docs:
                all_documents.extend(docs)
                self.processed_records[file_key] = current_hash
                newly_processed_files.append(file_path)
                print(f"  ✓ Loaded {len(docs)} document sections from {file_path.name}")

        if newly_processed_files:
            self._save_tracker()

        print(f"\nTotal new documents loaded for ingestion: {len(all_documents)}")
        return all_documents, newly_processed_files

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for RAG."""
        split_docs = self.splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        if split_docs:
            print("\nExample chunk:")
            print(f"Content: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")
        return split_docs