import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
import json
import os

# --------------------------
# 1. Generate a Synthetic Supply-Chain Graph
# --------------------------
def generate_supply_chain_graph(num_factories=5, num_warehouses=8, num_retailers=4):
    G = nx.DiGraph()

    # Define nodes with attributes
    # Factories
    for i in range(num_factories):
        G.add_node(f"Factory_{chr(65+i)}", 
                   type="factory",
                   capacity=np.random.randint(800, 1500),
                   location=(40.7 + np.random.normal(0, 0.1), -74.0 + np.random.normal(0, 0.1)),
                   processing_time=np.random.uniform(0.8, 1.2))

    # Warehouses
    for i in range(num_warehouses):
        G.add_node(f"Warehouse_{chr(88+i) if i < 3 else chr(65+i-3)}", 
                   type="warehouse",
                   storage=np.random.randint(400, 800),
                   location=(40.8 + np.random.normal(0, 0.1), -73.9 + np.random.normal(0, 0.1)),
                   processing_time=np.random.uniform(0.4, 0.6))

    # Retailers
    for i in range(num_retailers):
        G.add_node(f"Retailer_{i+1}",
                   type="retailer",
                   demand=np.random.randint(150, 300),
                   location=(40.6 + np.random.normal(0, 0.1), -73.7 + np.random.normal(0, 0.1)),
                   processing_time=np.random.uniform(0.1, 0.3))

    # Add edges
    # Connect factories to warehouses
    factories = [n for n, d in G.nodes(data=True) if d['type'] == 'factory']
    warehouses = [n for n, d in G.nodes(data=True) if d['type'] == 'warehouse']
    retailers = [n for n, d in G.nodes(data=True) if d['type'] == 'retailer']

    # Each factory connects to multiple warehouses
    for factory in factories:
        num_connections = np.random.randint(2, len(warehouses)-1)
        for warehouse in np.random.choice(warehouses, num_connections, replace=False):
            G.add_edge(factory, warehouse,
                      cost=np.random.uniform(4.0, 6.0),
                      lead_time=np.random.randint(1, 4),
                      capacity=np.random.randint(250, 400))

    # Each warehouse connects to multiple retailers
    for warehouse in warehouses:
        num_connections = np.random.randint(1, len(retailers))
        for retailer in np.random.choice(retailers, num_connections, replace=False):
            G.add_edge(warehouse, retailer,
                      cost=np.random.uniform(2.0, 4.0),
                      lead_time=np.random.uniform(0.5, 2.0),
                      capacity=np.random.randint(150, 250))

    return G

# --------------------------
# 2. Visualization and Complex Diagram Generation
# --------------------------
def plot_graph(G, layout='spring'):
    # Choose a layout for more complex diagrams:
    if layout == 'spring':
        # Adjust k (spacing) and iterations for better spacing
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G, scale=2)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, scale=2)
    else:
        # Fallback to positions using node location if available
        pos = {node: data.get("location", (np.random.random()*2, np.random.random()*2))
               for node, data in G.nodes(data=True)}

    # Draw nodes, coloring by type with different sizes
    node_colors = []
    node_sizes = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "factory":
            node_colors.append("#FF6B6B")  # Red
            node_sizes.append(1000)
        elif data.get("type") == "warehouse":
            node_colors.append("#4ECDC4")  # Cyan
            node_sizes.append(800)
        elif data.get("type") == "retailer":
            node_colors.append("#45B7D1")  # Blue
            node_sizes.append(600)
        else:
            node_colors.append("gray")
            node_sizes.append(500)

    plt.figure(figsize=(12, 8))
    
    # Draw edges with labels
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, width=1.5, alpha=0.6)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'cost')
    edge_labels = {k: f'Cost: {v:.1f}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9)
    
    # Add node labels with better formatting
    labels = {}
    for node in G.nodes():
        if 'Factory' in node:
            labels[node] = f'{node}\n(Cap: {G.nodes[node]["capacity"]})'
        elif 'Warehouse' in node:
            labels[node] = f'{node}\n(Store: {G.nodes[node]["storage"]})'
        elif 'Retailer' in node:
            labels[node] = f'{node}\n(Dem: {G.nodes[node]["demand"]})'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title("Supply Chain Network Diagram", pad=20, size=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --------------------------
# 3. Graph Analysis: Centrality & Bottleneck Detection
# --------------------------
def analyze_graph(G):
    # Compute node betweenness centrality to spot critical nodes.
    node_bt = nx.betweenness_centrality(G)
    print("Node Betweenness Centrality:")
    for node, score in node_bt.items():
        print(f"  {node}: {score:.3f}")

    # Compute edge betweenness centrality to identify bottleneck edges.
    edge_bt = nx.edge_betweenness_centrality(G)
    print("\nEdge Betweenness Centrality:")
    for edge, score in edge_bt.items():
        print(f"  {edge}: {score:.3f}")

# --------------------------
# 4. Prepare Data for GNN with PyTorch Geometric
# --------------------------
def create_pyg_data(G):
    # Map nodes to indices.
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

    # Create node feature matrix.
    features = []
    for node, data in G.nodes(data=True):
        # For each node, use one main attribute (capacity/storage/demand), latitude, longitude.
        if data.get("type") == "factory":
            feat = [data.get("capacity", 0), *data.get("location", (0, 0))]
        elif data.get("type") == "warehouse":
            feat = [data.get("storage", 0), *data.get("location", (0, 0))]
        elif data.get("type") == "retailer":
            feat = [data.get("demand", 0), *data.get("location", (0, 0))]
        else:
            feat = [0, 0, 0]
        features.append(feat)
    x = torch.tensor(features, dtype=torch.float)

    # Construct edge_index and edge_attr.
    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        # Features: [cost, lead_time, capacity]
        edge_attr.append([data.get("cost", 0), data.get("lead_time", 0), data.get("capacity", 0)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data

def save_graph(G, filename, format='json'):
    """Save the graph in specified format."""
    os.makedirs('saved_graphs', exist_ok=True)
    filepath = os.path.join('saved_graphs', filename)
    
    # Create a copy of the graph to modify for saving
    G_save = G.copy()
    
    # Convert tuples to strings for all nodes
    for node, data in G_save.nodes(data=True):
        if 'location' in data:
            data['location'] = f"{data['location'][0]},{data['location'][1]}"
    
    if format == 'json':
        # Convert graph to dictionary
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        for node, data in G_save.nodes(data=True):
            node_data = {'id': node}
            node_data.update(data)
            # Convert numpy values to native Python types
            for k, v in node_data.items():
                if isinstance(v, np.ndarray) or isinstance(v, np.number):
                    node_data[k] = v.item() if hasattr(v, 'item') else v.tolist()
            graph_data['nodes'].append(node_data)
        
        for u, v, data in G_save.edges(data=True):
            edge_data = {'source': u, 'target': v}
            edge_data.update(data)
            # Convert numpy values to native Python types
            for k, v in edge_data.items():
                if isinstance(v, np.ndarray) or isinstance(v, np.number):
                    edge_data[k] = v.item() if hasattr(v, 'item') else v.tolist()
            graph_data['edges'].append(edge_data)
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    elif format == 'graphml':
        nx.write_graphml(G_save, f"{filepath}.graphml")
    
    print(f"Graph saved as {filepath}.{format}")

def load_graph(filename, format='json'):
    """Load the graph from specified format."""
    filepath = os.path.join('saved_graphs', filename)
    
    if format == 'json':
        with open(f"{filepath}.json", 'r') as f:
            graph_data = json.load(f)
        
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_data in graph_data['nodes']:
            node_id = node_data.pop('id')
            # Convert location string back to tuple if it exists
            if 'location' in node_data:
                loc = node_data['location']
                if isinstance(loc, str):
                    node_data['location'] = tuple(float(x) for x in loc.split(','))
            G.add_node(node_id, **node_data)
        
        # Add edges with attributes
        for edge_data in graph_data['edges']:
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            G.add_edge(source, target, **edge_data)
    
    elif format == 'graphml':
        G = nx.read_graphml(f"{filepath}.graphml")
        # Convert location string back to tuple for all nodes
        for node, data in G.nodes(data=True):
            if 'location' in data:
                loc = data['location']
                if isinstance(loc, str):
                    data['location'] = tuple(float(x) for x in loc.split(','))
    
    return G


if __name__ == "__main__":
    # Generate the graph.
    G = generate_supply_chain_graph()

    # Save the graph in both formats
    save_graph(G, "supply_chain", format='json')
    save_graph(G, "supply_chain", format='graphml')

    # Load and visualize the saved graph
    loaded_G = load_graph("supply_chain", format='json')
    plot_graph(loaded_G, layout='spring')

    # Analyze the loaded graph
    analyze_graph(loaded_G)

    # Prepare and display the PyTorch Geometric data object.
    pyg_data = create_pyg_data(loaded_G)
    print("\nPyTorch Geometric Data Object:")
    print(pyg_data)
