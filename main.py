# get the SciQ dataset from the Hugginface
from datasets import load_dataset
import chromadb
from chromadb.config import Settings
from tqdm.notebook import tqdm
from umap.umap_ import UMAP
import umap.plot as umap_plot
import numpy as np


dataset = load_dataset("sciq", split="train")

# filter the dataset to only include questions with a support 
dataset = dataset.filter(lambda x: x['support'] != '')

print("Number of questions with support:", len(dataset))

chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="sciq")

# load the data and persist
collection.delete()

batch_size = 1000
for i in tqdm(range(0, len(dataset), batch_size)):
    collection.add(ids=[str(i) for i in range(i, min(i + batch_size, len(dataset)))], documents=dataset['support'][i:i + batch_size], metadatas=[{'type': 'support'} for _ in range(i, min(i + batch_size, len(dataset)))])

#get the embeddings for the support documents from the collection
support_embeddings = collection.get(include=['embeddings'])['embeddings']

dists = collection.query(
    query_embeddings=support_embeddings,
    n_results=2,
    include=['distances']
)

#flatten the distance list, excluding the first element (which is an element distance to itself)
flat_dists = [item for sublist in dists['distances'] for item in sublist[1:]]

mapper = UMAP().fit(support_embeddings)
umap_plot.points(mapper, values=np.array(flat_dists), show_legend=False, theme='inferno')

hist, bin_edges = np.histogram(flat_dists, bins=100, density=True)
cumulative_density = np.cumsum(hist) / np.sum(hist)

