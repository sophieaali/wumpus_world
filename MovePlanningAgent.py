from Environment import Environment
from Action import Action
from Percept import Percept
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import my_networkx as my_nx
import random


class MovePlanningAgent:
    
    def __init__(self, draw_graph=False, print_output=True):
        self.safe_locations = []
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
    

    
    def graph_safe_states(self, safe_locations):
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

    def choose_action(self):
        # return a randomly chosen Action except for Grab and Climb
        self.nextAction = Action(random.choice([0,1,2,4]))
        return self.nextAction
    
    def print_visualize_checks(self, env, percept, cumulative_reward):
        if self.print_output:
            print()
            print('Action:', self.nextAction)
            print()
            print('Agent Orientation:', env.agent_orientation)
            env.visualize()
            print('Percept:', percept)
            print('Cumulative reward:', cumulative_reward)
            print('-'*100)



    def run(self, WORLD_SIZE_x, WORLD_SIZE_y, allow_climb_without_gold, pit_prob):
        print("Environment Initialization:")
        env = Environment(WORLD_SIZE_x, WORLD_SIZE_y, allow_climb_without_gold, pit_prob)
        cumulative_reward = 0
        percept =  Percept(time_step=0, bump=False, breeze=env.is_breeze(), stench=env.is_stench(), scream=False, glitter=env.is_glitter(), reward=0, done=env.game_over)
        self.safe_locations.append((env.agent_location.x, env.agent_location.y))
        G = self.graph_safe_states(self.safe_locations) # Graph of safe locations
        env.visualize()
        print('Percept:', percept)
        print('Cumulative reward:', cumulative_reward)
        print('-'*100)

        while not percept.done:    
            self.choose_action()
            percept = env.step(self.nextAction)
            cumulative_reward += percept.reward
            self.print_visualize_checks(env, percept, cumulative_reward)
        
            # if the agent hasn't previously visited this location, add it to the set of safe locations and update the graph of safe locations
            loc_x, loc_y = env.agent_location.x, env.agent_location.y
            if (loc_x, loc_y) not in self.safe_locations:
                self.safe_locations.append((loc_x, loc_y))
                # Graph of safe locations
                G = self.graph_safe_states(self.safe_locations)

            # If the agent senses a glitter and does not yet have the gold, get the agent to automatically grab the gold as the next action
            if percept.glitter and env.agent_has_gold == False:
                self.nextAction = Action.GRAB
                percept = env.step(self.nextAction)
                cumulative_reward += percept.reward
                self.print_visualize_checks(env, percept, cumulative_reward)
                print()
                shortest_path, shortest_path_actions = self.find_shortest_path(G, f"({loc_x}, {loc_y}) {env.agent_orientation}", f"({1}, {1}) {env.agent_orientation}")
                print()
                print('-'*100)
                print()
                for action in shortest_path_actions:
                    if action == 'Forward':
                        self.nextAction = Action.FORWARD
                    elif action == 'TurnLeft':
                        self.nextAction = Action.LEFT
                    elif action == 'TurnRight':
                        self.nextAction = Action.RIGHT
                    
                    percept = env.step(self.nextAction)
                    cumulative_reward += percept.reward
                    self.print_visualize_checks(env, percept, cumulative_reward)

                # If the agent has the gold and is in square (1,1), the next action should be a climb
                self.nextAction = Action.CLIMB
                percept = env.step(self.nextAction)
                cumulative_reward += percept.reward
                self.print_visualize_checks(env, percept, cumulative_reward)
