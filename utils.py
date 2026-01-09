import os
import re
import uuid
import requests
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------- Qdrant setup ----------
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "wiki"
VECTOR_SIZE = 1024

if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


# ---------- Markdown utilities ----------
def read_markdown_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if content.lstrip().startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].lstrip("\n")

    return content


def normalize_text(s: str) -> str:
    return " ".join(s.split()).strip()


def clean_markdown(md: str) -> str:
    md = re.sub(r'(?s)```.*?```', ' ', md)
    md = re.sub(r'(?s)~~~.*?~~~', ' ', md)
    md = re.sub(r'`([^`]+)`', r'\1', md)
    md = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', md)
    md = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
    md = re.sub(r'(?m)^\s*#{1,6}\s*', '', md)
    md = re.sub(r'(?m)^\s*>\s*', '', md)
    md = re.sub(r'(?m)^\s*([-*+]\s+|\d+\.\s+)', '', md)
    md = re.sub(r'<[^>]+>', ' ', md)
    md = normalize_text(md)
    return md


def create_chunks(text: str, max_size: int = 500) -> list[str]:
    chunks = []
    current = ""

    for paragraph in text.split("\n\n"):
        if paragraph.startswith("#"):
            if current.strip():
                chunks.append(current.strip())
            current = paragraph
        else:
            if len(current) + len(paragraph) > max_size:
                chunks.append(current.strip())
                current = paragraph
            else:
                current += "\n\n" + paragraph if current else paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------- Embeddings ----------
def generate_embeddings(text: str) -> Optional[list[float]]:
    try:
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "mxbai-embed-large", "input": text},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception as e:
        print("Embedding failed:", e)
        return None


# ---------- Storage ----------
def store_article(chunks: list[str], source: str | None = None):
    for chunk in chunks:
        embedding = generate_embeddings(chunk)
        if embedding is None:
            continue

        payload = {"text": chunk}
        if source:
            base = os.path.splitext(os.path.basename(source))[0]
            base = re.sub(r'^\d+[_-]*', '', base)
            payload["source"] = base.replace("_", " ").strip()

        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )


# ---------- Generation ----------
def generate_response(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": False,
        },
    )
    return resp.json()["response"]