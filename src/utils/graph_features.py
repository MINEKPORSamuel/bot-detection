"""
Analyse Graphique - Projet Détection de Bots Twitter
Phase 1.4 : Construction du graphe d'interactions et calcul des features graphiques

Approche :
- Chaque utilisateur est un nœud du graphe
- Les arêtes (edges) représentent des interactions simulées
  à partir du Mention Count de chaque utilisateur
- 7 features graphiques calculées via NetworkX (équivalent à Spark GraphX) :
    1. degree_centrality
    2. in_degree_centrality
    3. out_degree_centrality
    4. closeness_centrality (estimée)
    5. betweenness_centrality (estimée)
    6. pagerank
    7. clustering_coefficient
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class GraphFeatureExtractor:
    """
    Construit un graphe d'interactions Twitter et calcule
    7 features graphiques pour chaque utilisateur.
    """

    def __init__(self,
                 raw_path='../bot_detection_data.csv',
                 processed_path='../data/processed_features.csv',
                 output_path='../data/graph_features.csv',
                 max_targets_per_user=25,
                 centrality_sample_size=30):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.output_path = output_path
        self.max_targets_per_user = max_targets_per_user
        self.centrality_sample_size = centrality_sample_size
        self.df_raw = None
        self.G = None
        self.df_graph = None

    def load_data(self):
        # Étape 1 : Chargement des données brutes depuis le CSV
        print("📥 Chargement des données...")
        self.df_raw = pd.read_csv(self.raw_path)
        print(f"✓ Dataset chargé : {self.df_raw.shape[0]} lignes × {self.df_raw.shape[1]} colonnes")
        return self.df_raw

    def build_graph(self):
        # Étape 2 : Construction du graphe d'interactions Twitter (les nœuds sont les comptes, les arêtes sont les mentions)
        print("\n🔗 Construction du graphe d'interactions...")

        self.G = nx.DiGraph()  # Graphe orienté (directed)

        # Ajouter tous les utilisateurs comme noeuds
        user_ids = self.df_raw['User ID'].tolist()
        self.G.add_nodes_from(user_ids)

        # Construire les arêtes à partir du Mention Count
        # Chaque utilisateur avec mention_count > 0 est connecté
        # à d'autres utilisateurs par ordre de follower count (simulation d'interactions)
        np.random.seed(42)
        df_sorted = self.df_raw.sort_values('Follower Count', ascending=False).reset_index(drop=True)
        top_users = df_sorted['User ID'].values
        top_201_candidates = list(top_users[:201])

        edges_added = 0
        for _, row in self.df_raw.iterrows():
            src = row['User ID']
            mention_count = int(row['Mention Count'])
            if mention_count == 0:
                continue

            if top_201_candidates[0] != src:
                targets = top_201_candidates[:200]
            else:
                targets = top_201_candidates[1:201]

            n_targets = min(mention_count, len(targets), self.max_targets_per_user)
            selected_targets = np.random.choice(targets, size=n_targets, replace=False)
            edge_weight = max(1.0, mention_count / n_targets)

            for tgt in selected_targets:
                self.G.add_edge(src, tgt, weight=edge_weight)
                edges_added += 1

        print(f"✓ Graphe construit :")
        print(f"   Noeuds : {self.G.number_of_nodes():,}")
        print(f"   Arêtes : {self.G.number_of_edges():,}")
        return self.G

    def _estimate_closeness(self, k=100):
        import random
        random.seed(42)
        n = self.G.number_of_nodes()
        nodes = list(self.G.nodes())
        samples = random.sample(nodes, min(k, n))
        
        G_rev = self.G.reverse()
        
        dist_sum = {node: 0.0 for node in self.G.nodes()}
        reachable_count = {node: 0 for node in self.G.nodes()}
        
        for s in samples:
            lengths = nx.single_source_shortest_path_length(G_rev, s)
            for v, dist in lengths.items():
                dist_sum[v] += dist
                reachable_count[v] += 1
                
        closeness = {}
        for v in self.G.nodes():
            if reachable_count[v] == 0 or dist_sum[v] == 0:
                closeness[v] = 0.0
            else:
                avg_dist = dist_sum[v] / reachable_count[v]
                reach_fraction = reachable_count[v] / len(samples)
                closeness[v] = reach_fraction / avg_dist
        return closeness

    # Étape 3 : Calcul des 7 caractéristiques (features) graphiques pour chaque utilisateur
    def compute_graph_features(self):
        print("\n⚙️  Calcul des 7 features graphiques...")

        # F1 : Degree Centrality (total — entrées + sorties)
        print("   1/7 Degree Centrality...")
        degree_cent = nx.degree_centrality(self.G)

        # F2 : In-Degree Centrality (combien de fois mentionné)
        print("   2/7 In-Degree Centrality...")
        in_degree_cent = nx.in_degree_centrality(self.G)

        # F3 : Out-Degree Centrality (combien de fois l'utilisateur mentionne)
        print("   3/7 Out-Degree Centrality...")
        out_degree_cent = nx.out_degree_centrality(self.G)

        # F4 : Closeness Centrality (approximate Eppstein-Wang)
        print(f"   4/7 Closeness Centrality (Eppstein-Wang k={self.centrality_sample_size})...")
        closeness_cent = self._estimate_closeness(k=self.centrality_sample_size)

        # F5 : Betweenness Centrality (approximate k=100)
        print(f"   5/7 Betweenness Centrality (k={self.centrality_sample_size})...")
        betweenness_cent = nx.betweenness_centrality(
            self.G,
            k=self.centrality_sample_size,
            normalized=True,
            seed=42
        )

        # F6 : PageRank (importance dans le réseau)
        print("   6/7 PageRank...")
        pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=100)

        # F7 : Clustering Coefficient (sur graphe non orienté)
        print("   7/7 Clustering Coefficient...")
        G_undirected = self.G.to_undirected()
        clustering = nx.clustering(G_undirected)

        # Assembler en DataFrame
        self.df_graph = pd.DataFrame({
            'User ID': list(degree_cent.keys()),
            'degree_centrality': list(degree_cent.values()),
            'in_degree_centrality': list(in_degree_cent.values()),
            'out_degree_centrality': list(out_degree_cent.values()),
            'closeness_centrality': [closeness_cent.get(node, 0.0) for node in degree_cent.keys()],
            'betweenness_centrality': [betweenness_cent.get(node, 0.0) for node in degree_cent.keys()],
            'pagerank': list(pagerank.values()),
            'clustering_coefficient': list(clustering.values()),
        })

        print(f"\n✓ 7 features graphiques calculées pour {len(self.df_graph):,} utilisateurs")
        return self.df_graph

    def merge_with_tabular(self):
        # Étape 4 : Jointure des caractéristiques graphiques obtenues avec les caractéristiques tabulaires existantes
        print("\n🔀 Fusion avec les features tabulaires (Phase 1.3)...")

        df_processed = pd.read_csv(self.processed_path)

        # Ajouter User ID au dataset processed pour la jointure
        df_raw_ids = self.df_raw[['User ID']].reset_index(drop=True)
        df_processed = df_processed.reset_index(drop=True)
        df_processed['User ID'] = df_raw_ids['User ID']

        # Fusion
        df_final = df_processed.merge(self.df_graph, on='User ID', how='left')
        df_final = df_final.drop(columns=['User ID'])

        # Remplir les éventuels NaN par 0
        graph_cols = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
                      'closeness_centrality', 'betweenness_centrality', 'pagerank', 'clustering_coefficient']
        df_final[graph_cols] = df_final[graph_cols].fillna(0)

        print(f"✓ Dataset final : {df_final.shape[0]:,} lignes × {df_final.shape[1]} colonnes")
        print(f"   Features tabulaires : 8")
        print(f"   Features graphiques : 7")
        print(f"   Colonne cible       : Bot Label")
        return df_final

    def process(self):
        # Point d'entrée pour exécuter l'ensemble du pipeline d'analyse graphique
        print("🚀 DEBUT DU PIPELINE D'ANALYSE GRAPHIQUE — Phase 1.4")

        self.load_data()
        self.build_graph()
        self.compute_graph_features()
        df_final = self.merge_with_tabular()

        # Sauvegarde
        df_final.to_csv(self.output_path, index=False)
        print(f"\n💾 Sauvegardé dans : {self.output_path}")
        print(f"   Dimensions : {df_final.shape}")

        # Résumé des features graphiques
        graph_cols = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
                      'closeness_centrality', 'betweenness_centrality', 'pagerank', 'clustering_coefficient']
        print("\n📊 Statistiques des features graphiques :")
        print(df_final[graph_cols].describe().round(6).to_string())

        print("\n✅ ANALYSE GRAPHIQUE TERMINEE AVEC SUCCES")
        print(f"Colonnes finales du dataset : {df_final.columns.tolist()}")

        return df_final


def main():
    extractor = GraphFeatureExtractor(
        raw_path='../bot_detection_data.csv',
        processed_path='../data/processed_features.csv',
        output_path='../data/graph_features.csv'
    )
    df_final = extractor.process()


if __name__ == "__main__":
    main()
