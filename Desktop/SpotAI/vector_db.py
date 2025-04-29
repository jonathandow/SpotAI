from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Configure index
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'spotai-tracks')
EMBED_DIM = int(os.getenv('EMBED_DIM', '1536'))
REGION = os.getenv('PINECONE_ENVIRONMENT')
spec = ServerlessSpec(cloud='aws', region=REGION)
existing = pc.list_indexes().names()
if INDEX_NAME not in existing:
    pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric='cosine', spec=spec)

# Connect to index
index = pc.Index(INDEX_NAME)

def upsert_embeddings(track_ids: list[str], embeddings: list[list[float]]):
    """
    Upsert track embeddings into Pinecone index.
    """
    # Validate inputs
    if not track_ids or not embeddings:
        print("No embeddings to upsert.")
        return
    if len(track_ids) != len(embeddings):
        print(f"Track IDs ({len(track_ids)}) and embeddings ({len(embeddings)}) length mismatch.")
        return
    # Prepare vector dicts
    vectors = [{"id": tid, "values": emb} for tid, emb in zip(track_ids, embeddings)]
    try:
        index.upsert(vectors=vectors)
    except Exception as e:
        print(f"Error upserting embeddings to Pinecone: {e}")

def query_similar(query_vector: list[float], top_k: int = 20) -> list[str]:
    """
    Query Pinecone for most similar track IDs.
    """
    # Validate inputs
    if not query_vector:
        print("Empty query vector provided.")
        return []
    
    # Ensure all values are valid floats
    for val in query_vector:
        if not isinstance(val, (int, float)) or float('nan') == val:
            print(f"Invalid value in query vector: {val}")
            return []
    
    try:
        res = index.query(vector=query_vector, top_k=top_k, include_values=False)
        return [match['id'] for match in res.get('matches', [])]
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []
