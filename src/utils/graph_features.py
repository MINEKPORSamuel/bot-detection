import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class GraphFeatureExtractor:
    def __init__(self, raw_path=None, processed_path=None, output_path=None):
        self.raw_path = raw_path or str(ROOT / 'bot_detection_data.csv')
        self.processed_path = processed_path or str(ROOT / 'data/processed_features.csv')
        self.output_path = output_path or str(ROOT / 'data/graph_features.csv')

    def process(self):
        print("[GRAPH FEATURES] ANALYSE DE GRAPHE...")
        df_raw = pd.read_csv(self.raw_path)
        df_tab = pd.read_csv(self.processed_path)
        
        G = nx.DiGraph()
        G.add_nodes_from(df_raw['User ID'])
        
        # Simulation d'arêtes (pour que PageRank ne soit pas nul)
        top_users = df_raw.nlargest(100, 'Follower Count')['User ID'].values
        for _, row in df_raw.iterrows():
            if row['Mention Count'] > 0:
                targets = np.random.choice(top_users, size=min(int(row['Mention Count']), 10), replace=False)
                for t in targets: G.add_edge(row['User ID'], t)
 
        # Calculs
        pg = nx.pagerank(G, alpha=0.85)
        dc = nx.degree_centrality(G)
        in_dc = nx.in_degree_centrality(G)
        out_dc = nx.out_degree_centrality(G)
        cl = nx.closeness_centrality(G)
        bt = nx.betweenness_centrality(G, k=50)
        cc = nx.clustering(G.to_undirected())
 
        df_graph = pd.DataFrame({
            'User ID': list(dc.keys()),
            'degree_centrality': list(dc.values()),
            'in_degree_centrality': list(in_dc.values()),
            'out_degree_centrality': list(out_dc.values()),
            'closeness_centrality': list(cl.values()),
            'betweenness_centrality': list(bt.values()),
            'pagerank': list(pg.values()),
            'clustering_coefficient': list(cc.values())
        })
 
        # Fusion
        df_tab['User ID'] = df_raw['User ID']
        df_final = df_tab.merge(df_graph, on='User ID').drop(columns=['User ID'])
        df_final.to_csv(self.output_path, index=False)
        print(f"[SUCCESS] Graphe terminé : {self.output_path}")
        return df_final
