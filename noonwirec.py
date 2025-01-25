from datetime import datetime
import chromadb
client = chromadb.Client()

collection = client.get_or_create_collection("test")

print(f"Collection created with name: {collection.name}")


collection.add(
    embeddings=[
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
    ],
    metadatas=[
        {"uri": "./imgs/stackoverflow-dev-survey-2024-technology-admired-and-desired-language-desire-admire-social.png", "style": "style1", "description": "My first chroma collection", "created": str(datetime.now())},
        {"uri": "./imgs/stackoverflow-dev-survey-2024-technology-admired-and-desired-language-desire-admire-social.png", "style": "style2", "description": "My first chroma collection", "created": str(datetime.now())},
        {"uri": "./imgs/stackoverflow-dev-survey-2024-technology-admired-and-desired-language-desire-admire-social.png", "style": "style1", "description": "My first chroma collection", "created": str(datetime.now())},
        {"uri": "./imgs/stackoverflow-dev-survey-2024-technology-admired-and-desired-language-desire-admire-social.png", "style": "style3", "description": "My first chroma collection", "created": str(datetime.now())},
        {"uri": "./imgs/stackoverflow-dev-survey-2024-technology-admired-and-desired-language-desire-admire-social.png", "style": "style1", "description": "My first chroma collection", "created": str(datetime.now())},
        {"uri": "./imgs/stackoverflow-dev-survey-2024-technology-admired-and-desired-language-desire-admire-social.png", "style": "style1", "description": "My first chroma collection", "created": str(datetime.now())},
        {"uri": "img7.png", "style": "style1"},
        {"uri": "img8.png", "style": "style1"},
    ],
    documents=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8"],
    ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
)

query_result = collection.query(
        query_embeddings=[[1.1, 2.3, 3.2], [5.1, 4.3, 2.2]],
        n_results=2,
    )

# print(query_result)
print(collection.peek())
