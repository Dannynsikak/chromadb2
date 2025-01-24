import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chromadb client 
client = chromadb.Client(Settings(
    persist_directory="./chroma_db" # directory to store the database
))

#TODO:2 create collection
# a collection in chromadb is like a table where you store related embeddings
article_collection = client.create_collection("articles")
user_collection = client.create_collection("users")

#TODO:3 store article embeddings
articles = [
    {"id": "1", "content": "Breaking news about AI", "tags": ["AI", "Technology"]},
    {"id": "2", "content": "Sports update: Football finals", "tags": ["Sports"]},
]

# generate embeddings for articles 
for article in articles:
    embedding = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).embeddings.create(input=article["content"], model="text-embedding-ada-002")['data'][0]['embedding']
    article_collection.add(
        ids=[article["id"]],
        embeddings=[embedding],
        metadatas=[{"tags": article["tags"], "content": article["content"]}]
    )


#TODO:4 store user interaction data
user_preferences = {
    "id": "user_1",
    "liked_topics": ["AI", "Technology"],
    "read_articles": ["1"], # Article id they have read
}

# convert preferences to an embedding
user_embedding = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).embeddings.create(input=" ".join(user_preferences["liked_topics"]),
model="text-embedding-ada-002")['data'][0]['embedding']

user_collection.add(
    ids=[user_preferences["id"]],
    embeddings=[user_embedding],
    metadatas=[user_preferences]
)

#TODO:5 perform a query
# to recommend articles based on user preferences
# retrive the user embedding
user = user_collection.get(ids=["user_1"])
user_embedding = user["embeddings"][0]

# find similar article 
results = article_collection.query(
    query_embeddings=[user_embedding],
    n_results=5 # top 5 recommendations
)

for match in results["metadatas"]:
    print("Recommended article:", match["content"])

# Example: User reads a new article
new_article = "Deep dive into AI ethics"
new_embedding = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).embeddings.create(input=new_article, model="text-embedding-ada-002")['data'][0]['embedding']

# Combine old embedding with new article embedding (e.g., by averaging)
updated_embedding = [
    (u + a) / 2 for u, a in zip(user_embedding, new_embedding)
]

# Update user collection
user_collection.update(
    ids=["user_1"],
    embeddings=[updated_embedding]
)
client.persist() # persist the changes