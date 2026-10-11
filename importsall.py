# Standard Library Imports
import os
import sys
import uuid
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

# Third-Party Scientific & Data Processing Imports
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings & Vector Database Imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# LangChain Document Schema & Loaders
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    TextLoader,
)

# LangChain Text Processing
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain LLM & Providers
from langchain_groq import ChatGroq

# Environment & Utilities
import requests
from dotenv import load_dotenv