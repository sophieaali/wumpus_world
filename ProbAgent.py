from Environment import Environment
from Action import Action
from Percept import Percept
from BayesianModelling import BuildBayesianNetwork
from MovePlanning import MovePlanning
from Location import Location
import random


class ProbAgent:
    
    def __init__(self, env, print_output=True):
        self.env = env
        self.print_output = print_output
        self.bayesian_model = BuildBayesianNetwork(self.env, self.print_output)
        # Build the initialized pits model
        # Build the initialized wumpus model

        self.safe_locations = {}
        self.stench_locations = []  # To track history of Stench locations
        self.breeze_locations = []  # To track history of Breeze locations

    def choose_action(self):
        # return a randomly chosen Action except for Grab and Climb
        self.nextAction = Action(random.choice([0,1,2,4]))
        return self.nextAction
    
    def print_visualize_checks(self, env, percept, cumulative_reward):
        print()
        print('Action:', self.nextAction)
        print()
        print('Agent Orientation:', env.agent_orientation)
        env.visualize()
        print('Percept:', percept)
        print('Cumulative reward:', cumulative_reward)
        print('-'*100)

    def go_to_11_and_climb(self, cumulative_reward):
        loc_x, loc_y = self.env.agent_location.x, self.env.agent_location.y
        G = MovePlanning(self.safe_locations.keys(), print_output=False).graph_safe_states()
        shortest_path, shortest_path_actions = MovePlanning(self.safe_locations.keys(), print_output=False).find_shortest_path(G, f"({loc_x}, {loc_y}) {self.env.agent_orientation}", f"({1}, {1}) {self.env.agent_orientation}")
        if self.print_output:
            print()
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
            
            percept = self.env.step(self.nextAction)
            cumulative_reward += percept.reward
            if self.print_output:
                self.print_visualize_checks(self.env, percept, cumulative_reward)

        self.nextAction = Action.CLIMB
        percept = self.env.step(self.nextAction)
        cumulative_reward += percept.reward
        if self.print_output:
            self.print_visualize_checks(self.env, percept, cumulative_reward)

        return percept, cumulative_reward
    

    def kill_wumpus_attempt(self, cumulative_reward):
        self.nextAction = Action.SHOOT
        percept = self.env.step(self.nextAction)
        cumulative_reward += percept.reward
        if self.print_output:
            self.print_visualize_checks(self.env, percept, cumulative_reward)

        return percept, cumulative_reward
        


    def run(self):
        cumulative_reward = 0
        percept =  Percept(time_step=0, bump=False, breeze=self.env.is_breeze(), stench=self.env.is_stench(), scream=False, glitter=self.env.is_glitter(), reward=0, done=self.env.game_over)
        self.safe_locations[(self.env.agent_location.x, self.env.agent_location.y)] = 1 # used to check if an agent  has 

        G = MovePlanning(self.safe_locations.keys(), print_output=False).graph_safe_states() # Graph of safe locations
        if self.print_output:
            print("Environment Initialization:")
            self.env.visualize()
            print('Percept:', percept)
            print('Cumulative reward:', cumulative_reward)
            print('-'*100)

        while not percept.done: 
            
            loc_x, loc_y = self.env.agent_location.x, self.env.agent_location.y   

            # if the game is not over yet
            # update pits and wumpus model based on new breeze percept (True or False)
            X_masked_pits = self.bayesian_model.update_pits_tensor()
            X_masked_wumpus = self.bayesian_model.update_stench_tensor(self.env.is_stench(), percept.scream)

            # Update pits and wumpus predictions
            new_wumpus_prediction = self.bayesian_model.get_new_wumpus_prediction(X_masked_wumpus)
            new_pits_predictions = self.bayesian_model.get_new_pits_predictions(X_masked_pits)
            
            # Get list of safe neighbours
            safe_neighbours, p_neighbours_pits, p_neighbours_wumpus = self.bayesian_model.get_safe_neighbours(new_pits_predictions, new_wumpus_prediction)

            # if any of the values in safe_neighbours_wumpus is equal to 0.5, we can attempt to kill the wumpus
            # if we hear a scream, we know the wumpus is dead and can ignore the wumpus model
            # if we don't hear a scream, we know exactly where the wumpus is and can update the wumpus model
            if any(value == 0.5 for value in p_neighbours_wumpus.values()):
                if self.print_output:
                    print("The agent can attempt to kill the wumpus")
                percept, cumulative_reward = self.kill_wumpus_attempt(cumulative_reward)
                if percept.scream:
                    if self.print_output:
                        print("Wumpus is dead - updating wumpus model")
                else:
                    if self.print_output:
                        print("Wumpus is not dead - updating wumpus model")
                    
                    # get the agent's location
                    agent_location = self.env.agent_location
                    # get the agent's orientation
                    agent_orientation = self.env.agent_orientation
                    # get the agent's neighbours
                    neighbours = Location(agent_location.x, agent_location.y).neighbours()
                    # get the location of the wumpus
                    no_wumpus_loc = Location.next_cell(agent_location, agent_orientation)
                    high_wumpus_probs = {key: value for key, value in p_neighbours_wumpus.items() if value == 0.5}
                    for key, value in high_wumpus_probs.items():
                        if Location(int(key[-2]), int(key[-1])) == no_wumpus_loc:
                            if self.print_output:
                                print("The wumpus is not at location:", (int(key[-2]), int(key[-1])))
                        else:
                            wumpus_loc = Location(int(key[-2]), int(key[-1]))
                            if self.print_output:
                                print("Wumpus Location:", wumpus_loc)
                    
                    # now update the wumpus model accordingly
                    # what is the index of the wumpus in the model
                    index = [Location(1, 1), Location(2, 1), Location(3, 1), Location(4, 1), Location(1, 2), Location(2, 2), Location(3, 2), Location(4, 2), Location(1, 3), Location(2, 3), Location(3, 3), Location(4, 3), Location(1, 4), Location(2, 4), Location(3, 4), Location(4, 4)].index(wumpus_loc)
                    X_masked_wumpus = self.bayesian_model.update_stench_tensor(self.env.is_stench, percept.scream, wumpus_loc_known = index)
                    # new_wumpus_prediction = self.bayesian_model.get_new_wumpus_prediction(X_masked_wumpus)
                    # # Get list of safe neighbours
                    # safe_neighbours, p_neighbours_pits, p_neighbours_wumpus = self.bayesian_model.get_safe_neighbours(new_pits_predictions, new_wumpus_prediction)
            
            # if there are no safe neighbours, 
            # or if an agent has visited any cell location at least 5 times (the agent is stuck in a loop)
            # proceed immediately to cell (1,1) and climb out
            elif len(safe_neighbours) == 0 or self.safe_locations[(loc_x, loc_y)] >= 5:
                if len(safe_neighbours) == 0:
                    if self.print_output:
                        print("There are no safe neighbours - proceeding to cell (1,1) and climbing out")
                    if (loc_x, loc_y) == (1,1):
                        if self.print_output:
                            print("The agent is already at cell (1,1) - climbing out")
                        self.nextAction = Action.CLIMB
                        percept = self.env.step(self.nextAction)
                        cumulative_reward += percept.reward
                        if self.print_output:
                            self.print_visualize_checks(self.env, percept, cumulative_reward)
                            print("The percept done is:", percept.done)
                            print("Game Over")
                        return cumulative_reward
                    
                if self.safe_locations[(loc_x, loc_y)] >= 5:
                    if self.print_output:
                        print("The agent is stuck in a loop - proceeding to cell (1,1) and climbing out")
                    if self.env.agent_location.x == 1 and self.env.agent_location.y == 1:
                        
                        self.nextAction = Action.CLIMB
                        percept = self.env.step(self.nextAction)
                        cumulative_reward += percept.reward
                        if self.print_output:
                            print("The agent is already at cell (1,1) - climbing out")
                            self.print_visualize_checks(self.env, percept, cumulative_reward)
                            print("The percept done is:", percept.done)
                            print("Game Over")
                        return cumulative_reward
                    
                
                    percept, cumulative_reward = self.go_to_11_and_climb(cumulative_reward)
                    if self.print_output:
                        print("The percept done is:", percept.done)
                        print("Game Over")

                    return cumulative_reward
            
            else:
                # safe neighbours exist (the game can continue) - what is the quickest path to get to a safe neighbour
                fastest_safe_move_location, fastest_safe_move_path = self.bayesian_model.get_quickest_path(safe_neighbours, self.safe_locations.keys())

                for action in fastest_safe_move_path:
                    if action == 'Forward':
                        self.nextAction = Action.FORWARD
                    elif action == 'TurnLeft':
                        self.nextAction = Action.LEFT
                    elif action == 'TurnRight':
                        self.nextAction = Action.RIGHT
                    
                    percept = self.env.step(self.nextAction)
                    cumulative_reward += percept.reward
                    if self.print_output:
                        self.print_visualize_checks(self.env, percept, cumulative_reward)

                # if the agent hasn't previously visited this location, add it to the set of safe locations and update the graph of safe locations
                loc_x, loc_y = self.env.agent_location.x, self.env.agent_location.y
                if (loc_x, loc_y) not in self.safe_locations.keys():
                    self.safe_locations[(loc_x, loc_y)] = 1
                else:
                    self.safe_locations[(loc_x, loc_y)] += 1
            
                # If the agent senses a glitter and does not yet have the gold, get the agent to automatically grab the gold as the next action
                if percept.glitter and self.env.agent_has_gold == False:
                    self.nextAction = Action.GRAB
                    percept = self.env.step(self.nextAction) # grab gold
                    cumulative_reward += percept.reward
                    if self.print_output:
                        print("AGENT HAS GOLD - PROCEEDING TO CELL (1,1) AND CLIMBING OUT")
                    percept, cumulative_reward = self.go_to_11_and_climb(cumulative_reward)
                    if self.print_output:
                        self.print_visualize_checks(self.env, percept, cumulative_reward)
                        # graph safe states and proceed immediately to cell (1,1) and climb out
                        print("The percept done is:", percept.done)
                        print("Game Over")
        return cumulative_reward

