import networkx as nx

G = nx.cycle_graph(5)
centrality = nx.eigenvector_centrality(G)
print(centrality)
