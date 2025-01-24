import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and general-purpose embedding model

# Initialize chromadb client 
client = chromadb.Client(Settings(
    persist_directory="./chroma_db"  # Directory to store the database
))

# TODO:2 Create collections
# A collection in chromadb is like a table where you store related embeddings
article_collection = client.create_collection("articles")
user_collection = client.create_collection("users")

# TODO:3 Store article embeddings
articles = [
    {"id": "1", "content": "Breaking news about AI", "tags": ["AI", "Technology"]},
    {"id": "2", "content": "Sports update: Football finals", "tags": ["Sports"]},
]

# Generate embeddings for articles
for article in articles:
    embedding = model.encode(article["content"]).tolist()  # Generate and convert embeddings to a list
    article_collection.add(
        ids=[article["id"]],
        embeddings=[embedding],
        metadatas=[{"tags": article["tags"], "content": article["content"]}]
    )

# TODO:4 Store user interaction data
user_preferences = {
    "id": "user_1",
    "liked_topics": ["AI", "Technology"],
    "read_articles": ["1"],  # Article ID they have read
}

# Convert preferences to an embedding
user_embedding = model.encode(" ".join(user_preferences["liked_topics"])).tolist()

user_collection.add(
    ids=[user_preferences["id"]],
    embeddings=[user_embedding],
    metadatas=[user_preferences]
)

# TODO:5 Perform a query
# Retrieve the user embedding
user = user_collection.get(ids=["user_1"])
user_embedding = user["embeddings"][0]

# Find similar articles
results = article_collection.query(
    query_embeddings=[user_embedding],
    n_results=5  # Top 5 recommendations
)

for match in results["metadatas"]:
    print("Recommended article:", match["content"])

# Example: User reads a new article
new_article = "Deep dive into AI ethics"
new_embedding = model.encode(new_article).tolist()

# Combine old embedding with new article embedding (e.g., by averaging)
updated_embedding = [
    (u + a) / 2 for u, a in zip(user_embedding, new_embedding)
]

# Update user collection
user_collection.update(
    ids=["user_1"],
    embeddings=[updated_embedding]
)

# Persist the changes
client.persist()
