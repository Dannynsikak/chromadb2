from datetime import datetime
from chromadb import Client
from chromadb.config import Settings

# Initialize ChromaDB client
client = Client(Settings())

# Define collections based on schema
collections = {
    "users": client.create_collection(name="users"),
    "articles": client.create_collection(name="articles"),
    "tags": client.create_collection(name="tags"),
    "categories": client.create_collection(name="categories"),
    "article_tags": client.create_collection(name="article_tags"),
    "article_categories": client.create_collection(name="article_categories"),
    "article_views": client.create_collection(name="article_views"),
    "article_likes": client.create_collection(name="article_likes"),
    "article_saves": client.create_collection(name="article_saves"),
    "article_shares": client.create_collection(name="article_shares"),
}
print(f"Created {len(collections)} collections.")
# Add sample data to collections
# Users collection
collections["users"].add(
    ids=["1"],
    embeddings=[[0.1, 0.2, 0.3]],  # Placeholder for embeddings
    metadatas=[
        {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "role": "admin",
            "registered_at": "2025-01-01T10:00:00",
        }
    ],
)
print("Added sample user data for", collections["users"].name)
# Articles collection
collections["articles"].add(
    ids=["101"],
    embeddings=[[0.4, 0.5, 0.6]],  # Embedding of the article content
    metadatas=[
        {
            "author_id": "1",
            "title": "Understanding ChromaDB",
            "slug": "understanding-chromadb",
            "summary": "A beginner's guide to ChromaDB.",
            "content": "This is a detailed article on ChromaDB...",
            "created_at": str(datetime.now())
        }
    ],
)
print("Added sample article data for", collections["articles"].name)

# Tags collection
collections["tags"].add(
    ids=["301"],
    embeddings=[[0.2, 0.3, 0.4]],  # Embedding for tag description
    metadatas=[
        {"name": "Tech", "slug": "tech", "meta_title": "Technology News"}
    ],
)
print("Added sample tag data for", collections["tags"].name)
# Relationships (e.g., Article Tags)
collections["article_tags"].add(
    ids=["1001"],
    embeddings=[[0.0, 0.0, 0.0]],  # Placeholder, as this is a relationship
    metadatas=[
        {"article_id": "101", "tag_id": "301"}
    ],
)
print("Added sample relationship data for", collections["article_tags"].name)
