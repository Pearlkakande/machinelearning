import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load artifacts
@st.cache_data
def load_artifacts():
    df    = pd.read_csv('books_data.csv')
    emb   = pickle.load(open('book_embeddings.pkl','rb'))
    graph = pickle.load(open('book_graph.pkl','rb'))
    state = torch.load('gnn_model.pt', map_location='cpu')
    return df, emb, graph, state

# Define GNN (matching 384→384)
class GNNRecommender(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, in_ch)
        self.conv2 = GCNConv(in_ch, in_ch)
    def encode(self, data):
        x  = data['book'].x
        ei = data['book','similar_to','book'].edge_index
        x  = F.relu(self.conv1(x, ei))
        return self.conv2(x, ei)

# Recommendation logic
def recommend_with_gnn(genre, desc, df, emb, graph, state, top_n=5):
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb    = st_model.encode([f"{genre} {desc}"])[0]
    # Build and load GNN
    model = GNNRecommender(graph['book'].x.size(1))
    model.load_state_dict(state)
    model.eval()
    # Encode books
    with torch.no_grad():
        book_embs = model.encode(graph).cpu().numpy()
    book_embs = np.nan_to_num(book_embs)
    sims      = cosine_similarity([q_emb], book_embs)[0]
    idx       = np.argsort(sims)[-top_n:][::-1]
    recs      = df.iloc[idx].copy()
    recs['Score'] = sims[idx]
    return recs[['Book','Author','Genres','Score']]

# Streamlit UI
def main():
    st.title("📚 Hybrid Book Recommender")
    df, emb, graph, state = load_artifacts()
    genre = st.text_input("Genre", "Science Fiction")
    desc  = st.text_area("Description",
                         "space exploration and alien civilizations")
    if st.button("Recommend"):
        recs = recommend_with_gnn(genre, desc, df, emb, graph, state, top_n=3)
        st.dataframe(recs)


main()
