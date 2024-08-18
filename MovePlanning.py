import networkx as nx
import matplotlib.pyplot as plt
import my_networkx as my_nx

class MovePlanning:

    def __init__(self, safe_locations, draw_graph=False, print_output=True):
        self.safe_locations = safe_locations    
        self.draw_graph = draw_graph
        self.print_output = print_output

    def create_edge_list(self, points):
        """
        Create a list of edges for a graph of safe locations.
        Each edge represents a possible action the agent can take from a given location and direction.
        """
        directions = ['N', 'W', 'S', 'E']
        turn_actions = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
        edge_list = []

        for point in points:
            x, y = point
            for dir in directions:
                next_dir = turn_actions[dir]
                # Turning left
                edge_list.append((f"{point} {dir}", f"{point} {next_dir}", {'w': 'TurnLeft'}))
                # Turning right (reverse the turn_actions mapping)
                right_turn_dir = {v: k for k, v in turn_actions.items()}[dir]
                edge_list.append((f"{point} {dir}", f"{point} {right_turn_dir}", {'w': 'TurnRight'}))

        # Calculate new point for forward movement
                if dir == 'E':
                    new_point = (x + 1, y)
                elif dir == 'W':
                    new_point = (x - 1, y)
                elif dir == 'N':
                    new_point = (x, y + 1)
                else:  # dir == 'S'
                    new_point = (x, y - 1)
                
                # Add forward movement if new point exists in points
                if new_point in points:
                    edge_list.append((f"{point} {dir}", f"{new_point} {dir}", {'w': 'Forward'}))

        return edge_list
    
    def manhattan_distance(self, node1, node2):
        """
        Calculate the Manhattan distance between two nodes.
        Nodes are in the format '(x,y) D' where x and y are coordinates and D is a direction.
        """
        # Extract coordinates from the node labels
        x1, y1 = map(int, node1[1:node1.find(')')].split(','))
        x2, y2 = map(int, node2[1:node2.find(')')].split(','))
        
        return abs(x1 - x2) + abs(y1 - y2)
    
    
    def get_edge_labels_for_path(self, graph, path):
        """
        Get the edge labels for the shortest path in a graph.
        """
        edge_labels = {}
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1])
            # Assuming the edge label or weight is stored under the key 'weight'
            # Adjust the key as necessary based on your graph's edge attributes
            edge_label = edge_data.get('w', 'N/A')  # Default to 'N/A' if not found
            edge_labels[(path[i], path[i+1])] = edge_label
        return edge_labels
    
    
    def graph_safe_states(self):
        """
        Create a graph of safe states.
        """
        G = nx.DiGraph()

        points = self.safe_locations
        # Create the edge list'
        edge_list = self.create_edge_list(points)
        G.add_edges_from(edge_list)

        edge_weights = nx.get_edge_attributes(G,'w')
        edge_labels = {edge: edge_weights[edge] for edge in G.edges()}

        if self.draw_graph:
            pos=nx.spring_layout(G,seed=5)
            fig, ax = plt.subplots()
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=450)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=5)
            arc_rad = 0.25
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=G.edges(), connectionstyle=f'arc3, rad = {arc_rad}', arrowsize=20)
            
            my_nx.my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,rotate=False,rad = arc_rad, font_size=5)

        return G
    
    def find_shortest_path(self, G, start_node, end_node):
        """
        Find the shortest path between two nodes in a graph.
        """
        shortest_path = nx.astar_path(G, start_node, end_node, heuristic=self.manhattan_distance)
        # Get edge labels for the shortest path
        edge_labels = self.get_edge_labels_for_path(G, shortest_path)
        shortest_path_actions = list(edge_labels.values())

        # Remove any turns at the end of the shortest_path_actions
        while shortest_path_actions[-1] in ['TurnLeft', 'TurnRight']:
            shortest_path_actions.pop()
            shortest_path.pop()

        # Print shortest path and edge labels
        if self.print_output:
            print("Edge labels for the shortest path:", shortest_path_actions)

        return shortest_path, shortest_path_actions
                