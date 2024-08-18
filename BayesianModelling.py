from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch
from Location import Location
from Environment import Environment
from MovePlanningAgent import MovePlanningAgent
from StenchCategoricalProbabilities import StenchCategoricalProbabilities
import math


class BuildBayesianNetwork:

    def __init__(self, env, print_output=True):

        self.print_output = print_output

        self.mod_pit11 = 0
        self.mod_pit21 = -1
        self.mod_pit31 = -1
        self.mod_pit41 = -1
        self.mod_pit12 = -1
        self.mod_pit22 = -1
        self.mod_pit32 = -1
        self.mod_pit42 = -1
        self.mod_pit13 = -1
        self.mod_pit23 = -1
        self.mod_pit33 = -1
        self.mod_pit43 = -1
        self.mod_pit14 = -1
        self.mod_pit24 = -1
        self.mod_pit34 = -1
        self.mod_pit44 = -1
        self.mod_breeze11 = -1
        self.mod_breeze21 = -1
        self.mod_breeze31 = -1
        self.mod_breeze41 = -1
        self.mod_breeze12 = -1
        self.mod_breeze22 = -1
        self.mod_breeze32 = -1
        self.mod_breeze42 = -1
        self.mod_breeze13 = -1
        self.mod_breeze23 = -1
        self.mod_breeze33 = -1
        self.mod_breeze43 = -1
        self.mod_breeze14 = -1
        self.mod_breeze24 = -1
        self.mod_breeze34 = -1
        self.mod_breeze44 = -1

        self.full_pits_model = self.build_pits_bayesian_model()
        self.env = env
        
        self.wumpus = None
        self.tensor_wumpus = -1
        self.tensor_stench11 = -1
        self.tensor_stench21 = -1
        self.tensor_stench31 = -1
        self.tensor_stench41 = -1
        self.tensor_stench12 = -1
        self.tensor_stench22 = -1
        self.tensor_stench32 = -1
        self.tensor_stench42 = -1
        self.tensor_stench13 = -1
        self.tensor_stench23 = -1
        self.tensor_stench33 = -1
        self.tensor_stench43 = -1
        self.tensor_stench14 = -1
        self.tensor_stench24 = -1
        self.tensor_stench34 = -1
        self.tensor_stench44 = -1

        self.wumpus_model = self.build_wumpus_bayesian_model()

    class Predicate:
        def __init__(self, prob: float):
            self.p = prob            
            
        def toList(self):
            return [1-self.p, self.p]
            
        def toCategorical(self):
            return Categorical([self.toList()])
        
    def probabilities_of_pits(self, p=0.2):
        self.pit11 = self.Predicate(0).toCategorical()
        self.pit21 = self.Predicate(p).toCategorical()
        self.pit31 = self.Predicate(p).toCategorical()
        self.pit41 = self.Predicate(p).toCategorical()
        self.pit12 = self.Predicate(p).toCategorical()
        self.pit22 = self.Predicate(p).toCategorical()
        self.pit32 = self.Predicate(p).toCategorical()
        self.pit42 = self.Predicate(p).toCategorical()
        self.pit13 = self.Predicate(p).toCategorical()
        self.pit23 = self.Predicate(p).toCategorical()
        self.pit33 = self.Predicate(p).toCategorical()
        self.pit43 = self.Predicate(p).toCategorical()
        self.pit14 = self.Predicate(p).toCategorical()
        self.pit24 = self.Predicate(p).toCategorical()
        self.pit34 = self.Predicate(p).toCategorical()
        self.pit44 = self.Predicate(p).toCategorical()

    def probabilities_of_wumpus(self):
        self.wumpus = Categorical([[0, 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15., 1./15.]])

    def get_cat_dist_cell_2_parents(self, parent1, parent2):

        cases = []
        for parent1 in [False, True]:
            c = []
            for parent2 in [False, True]:
                case = parent1 or parent2 # we can use or between them because we have a breeze if there is a pit in any of the adjacent squares
                if case:
                    p = 1.0
                else:
                    p = 0.0
                c.append(self.Predicate(p).toList())
            cases.append(c)
        cell = ConditionalCategorical([cases])
        return cell
    
    def get_cat_dist_cell_3_parents(self, parent1, parent2, parent3):
        cases = []
        for parent1 in [False, True]:
            b = []
            for parent2 in [False, True]:
                c = []
                for parent3 in [False, True]:
                    case = parent1 or parent2 or parent3
                    if case:
                        p = 1.0
                    else:
                        p = 0.0
                    c.append(self.Predicate(p).toList())
                b.append(c)
            cases.append(b)
        cell = ConditionalCategorical([cases])
        return cell
    
    def get_cat_dist_cell_4_parents(self, parent1, parent2, parent3, parent4):
        cases = []
        for parent1 in [False, True]:
            a = []
            for parent2 in [False, True]:
                b = []
                for parent3 in [False, True]:
                    c = []
                    for parent4 in [False, True]:
                        case = parent1 or parent2 or parent3 or parent4
                        if case:
                            p = 1.0
                        else:
                            p = 0.0
                        c.append(self.Predicate(p).toList())
                    b.append(c)
                a.append(b)
            cases.append(a)
        cell = ConditionalCategorical([cases])
        return cell

    def get_cat_dist_of_breezes(self):
        self.breeze11 = self.get_cat_dist_cell_2_parents(self.pit12, self.pit21)
        self.breeze41 = self.get_cat_dist_cell_2_parents(self.pit31, self.pit42)
        self.breeze14 = self.get_cat_dist_cell_2_parents(self.pit13, self.pit24)
        self.breeze44 = self.get_cat_dist_cell_2_parents(self.pit43, self.pit34)
        self.breeze12 = self.get_cat_dist_cell_3_parents(self.pit11, self.pit22, self.pit13)
        self.breeze21 = self.get_cat_dist_cell_3_parents(self.pit11, self.pit22, self.pit31)
        self.breeze31 = self.get_cat_dist_cell_3_parents(self.pit21, self.pit32, self.pit41)
        self.breeze13 = self.get_cat_dist_cell_3_parents(self.pit12, self.pit23, self.pit14)
        self.breeze24 = self.get_cat_dist_cell_3_parents(self.pit14, self.pit23, self.pit34)
        self.breeze34 = self.get_cat_dist_cell_3_parents(self.pit24, self.pit33, self.pit44)
        self.breeze43 = self.get_cat_dist_cell_3_parents(self.pit42, self.pit33, self.pit44)
        self.breeze42 = self.get_cat_dist_cell_3_parents(self.pit41, self.pit32, self.pit43)
        self.breeze22 = self.get_cat_dist_cell_4_parents(self.pit12, self.pit21, self.pit23, self.pit32)
        self.breeze32 = self.get_cat_dist_cell_4_parents(self.pit22, self.pit31, self.pit33, self.pit42) 
        self.breeze23 = self.get_cat_dist_cell_4_parents(self.pit13, self.pit22, self.pit24, self.pit33)
        self.breeze33 = self.get_cat_dist_cell_4_parents(self.pit23, self.pit32, self.pit34, self.pit43)

    def get_cat_dist_of_stenches(self):
        stenchcategoricaldistributions = StenchCategoricalProbabilities()
        stenchcategoricaldistributions.create_stench_probabilities()

        self.stench11 = stenchcategoricaldistributions.stench11
        self.stench21 = stenchcategoricaldistributions.stench21
        self.stench31 = stenchcategoricaldistributions.stench31
        self.stench41 = stenchcategoricaldistributions.stench41

        self.stench12 = stenchcategoricaldistributions.stench12
        self.stench22 = stenchcategoricaldistributions.stench22
        self.stench32 = stenchcategoricaldistributions.stench32
        self.stench42 = stenchcategoricaldistributions.stench42

        self.stench13 = stenchcategoricaldistributions.stench13
        self.stench23 = stenchcategoricaldistributions.stench23
        self.stench33 = stenchcategoricaldistributions.stench33
        self.stench43 = stenchcategoricaldistributions.stench43

        self.stench14 = stenchcategoricaldistributions.stench14
        self.stench24 = stenchcategoricaldistributions.stench24
        self.stench34 = stenchcategoricaldistributions.stench34
        self.stench44 = stenchcategoricaldistributions.stench44


    def build_pits_bayesian_model(self):

        self.probabilities_of_pits()
        self.get_cat_dist_of_breezes()

        variables = [self.pit11, self.pit21, self.pit31, self.pit41, self.pit12, self.pit22, self.pit32, self.pit42, self.pit13, self.pit23, self.pit33, self.pit43, self.pit14, self.pit24, self.pit34, self.pit44, 
                  self.breeze11, self.breeze21, self.breeze31, self.breeze41, self.breeze12, self.breeze22, self.breeze32, self.breeze42, self.breeze13, self.breeze23, self.breeze33, self.breeze43, self.breeze14, self.breeze24, self.breeze34, self.breeze44]

        edges = [(self.pit12, self.breeze11), (self.pit21, self.breeze11), (self.pit31, self.breeze41), (self.pit42, self.breeze41), (self.pit13, self.breeze14), (self.pit24, self.breeze14), 
                    (self.pit34, self.breeze44), (self.pit43, self.breeze44), 
                    (self.pit11, self.breeze21), (self.pit22, self.breeze21), (self.pit31, self.breeze21), (self.pit21, self.breeze31), (self.pit32, self.breeze31), (self.pit41, self.breeze31),
                    (self.pit11, self.breeze12), (self.pit22, self.breeze12), (self.pit13, self.breeze12), (self.pit12, self.breeze13), (self.pit23, self.breeze13), (self.pit14, self.breeze13), 
                    (self.pit14, self.breeze24), (self.pit23, self.breeze24), (self.pit34, self.breeze24), (self.pit24, self.breeze34), (self.pit33, self.breeze34), (self.pit44, self.breeze34), 
                    (self.pit44, self.breeze43), (self.pit33, self.breeze43), (self.pit42, self.breeze43), (self.pit43, self.breeze42), (self.pit32, self.breeze42), (self.pit41, self.breeze42),
                    (self.pit21, self.breeze22), (self.pit12, self.breeze22), (self.pit23, self.breeze22), (self.pit32, self.breeze22), (self.pit31, self.breeze32), (self.pit22, self.breeze32),
                    (self.pit33, self.breeze32), (self.pit42, self.breeze32), (self.pit22, self.breeze23), (self.pit13, self.breeze23), (self.pit24, self.breeze23), (self.pit33, self.breeze23),
                    (self.pit32, self.breeze33), (self.pit23, self.breeze33), (self.pit34, self.breeze33), (self.pit43, self.breeze33)]

        pits_model = BayesianNetwork(variables, edges)
        return pits_model
    
    def build_wumpus_bayesian_model(self):

        self.probabilities_of_wumpus()
        self.get_cat_dist_of_stenches()

        variables = [self.wumpus, self.stench11, self.stench21, self.stench31, self.stench41, self.stench12, self.stench22, self.stench32, self.stench42, self.stench13, self.stench23, self.stench33, self.stench43, self.stench14, self.stench24, self.stench34, self.stench44]
        edges = [(self.wumpus, self.stench11), (self.wumpus, self.stench21), (self.wumpus, self.stench31), (self.wumpus, self.stench41), (self.wumpus, self.stench12), (self.wumpus, self.stench22), (self.wumpus, self.stench32), (self.wumpus, self.stench42), (self.wumpus, self.stench13), (self.wumpus, self.stench23), (self.wumpus, self.stench33), (self.wumpus, self.stench43), (self.wumpus, self.stench14), (self.wumpus, self.stench24), (self.wumpus, self.stench34), (self.wumpus, self.stench44)]

        wumpus_model = BayesianNetwork(variables, edges)

        return wumpus_model
    

    def update_pits_tensor(self):
        
        if not self.env.game_over:
            loc_x, loc_y = self.env.agent_location.x, self.env.agent_location.y
            # Generate the variable name
            variable_name = f'self.mod_pit{loc_x}{loc_y}'
            # Update the model 
            exec(f'{variable_name} = 0')


            if self.env.is_breeze(): # if a breeze is sensed
                # Generate the variable name
                variable_name = f'self.mod_breeze{loc_x}{loc_y}'

                # Update the model 
                exec(f'{variable_name} = 1')

            else: # if no breeze is sensed
                # Generate the variable name
                variable_name = f'self.mod_breeze{loc_x}{loc_y}'

                # Update the model 
                exec(f'{variable_name} = 0')

         # update the pits model with the new percept information
        new_torch_tensor = [[self.mod_pit11, self.mod_pit21, self.mod_pit31, self.mod_pit41, self.mod_pit12, self.mod_pit22, self.mod_pit32, self.mod_pit42, self.mod_pit13, self.mod_pit23, self.mod_pit33, self.mod_pit43, self.mod_pit14, self.mod_pit24, self.mod_pit34, self.mod_pit44,
        self.mod_breeze11, self.mod_breeze21, self.mod_breeze31, self.mod_breeze41, self.mod_breeze12, self.mod_breeze22, self.mod_breeze32, self.mod_breeze42, self.mod_breeze13, self.mod_breeze23, self.mod_breeze33, self.mod_breeze43, self.mod_breeze14, self.mod_breeze24, self.mod_breeze34, self.mod_breeze44]]

        X = torch.tensor(new_torch_tensor)
        X_masked = torch.masked.MaskedTensor(X, mask=X >= 0)

        return X_masked
    

    def update_stench_tensor(self, stench_percept, scream_percept, wumpus_loc_known = False):
        if wumpus_loc_known:
            self.tensor_wumpus = wumpus_loc_known
        if scream_percept: # if the wumpus was killed
            # new_torch_tensor = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
            self.tensor_wumpus = 0
        
        else: # Wumpus is still alive, update the model with the new percept information
            loc_x, loc_y = self.env.agent_location.x, self.env.agent_location.y

            if stench_percept:  # if a stench is sensed
                # Generate the variable name
                variable_name = f'self.tensor_stench{loc_x}{loc_y}'

                # Update the model
                exec(f'{variable_name} = 1')

            else:  # if no stench is sensed
                # Generate the variable name
                variable_name = f'self.tensor_stench{loc_x}{loc_y}'
                # Update the model
                exec(f'{variable_name} = 0')

                # update the wumpus model with the new percept information
        new_torch_tensor =  [[self.tensor_wumpus, self.tensor_stench11, self.tensor_stench21, self.tensor_stench31, self.tensor_stench41, self.tensor_stench12, self.tensor_stench22, self.tensor_stench32, self.tensor_stench42, self.tensor_stench13, self.tensor_stench23, self.tensor_stench33, self.tensor_stench43, self.tensor_stench14, self.tensor_stench24, self.tensor_stench34, self.tensor_stench44]]
        X = torch.tensor(new_torch_tensor)
    
        X_masked = torch.masked.MaskedTensor(X, mask=X >= 0)

        return X_masked

    def get_new_pits_predictions(self, X_masked):
        new_pits_predictions = self.full_pits_model.predict_proba(X_masked)
        return new_pits_predictions
    
    def get_new_wumpus_prediction(self, X_masked):
        new_wumpus_prediction = self.wumpus_model.predict_proba(X_masked)
        return new_wumpus_prediction

    def get_safe_neighbours(self, new_pits_predictions, new_wumpus_prediction):

        # new_pits_predictions = self.get_new_pits_predictions(X_masked, self.full_pits_model)
        # get the neighbours of the current location
        neighbours = Location(self.env.agent_location.x, self.env.agent_location.y).neighbours()
        # get probabilities each neighbour cell is safe
        p_neighbours_pits = {}
        p_neighbours_wumpus = {}
        pit_variables = ['pit11', 'pit21', 'pit31', 'pit41', 'pit12', 'pit22', 'pit32', 'pit42', 'pit13', 'pit23', 'pit33', 'pit43', 'pit14', 'pit24', 'pit34', 'pit44', 
                    'breeze11', 'breeze21', 'breeze31', 'breeze41', 'breeze12', 'breeze22', 'breeze32', 'breeze42', 'breeze13', 'breeze23', 'breeze33', 'breeze43', 'breeze14', 'breeze24', 'breeze34', 'breeze44']
        wumpus_variables = ['neighbour11', 'neighbour21', 'neighbour31', 'neighbour41', 'neighbour12', 'neighbour22', 'neighbour32', 'neighbour42', 'neighbour13', 'neighbour23', 'neighbour33', 'neighbour43', 'neighbour14', 'neighbour24', 'neighbour34', 'neighbour44']
        
        for neighbour in neighbours:
            index = pit_variables.index(f'pit{neighbour.x}{neighbour.y}')
            p_neighbours_pits[f'pit{neighbour.x}{neighbour.y}'] = new_pits_predictions[index][0][0].item()

            index = wumpus_variables.index(f'neighbour{neighbour.x}{neighbour.y}')
            p_neighbours_wumpus[f'neighbour{neighbour.x}{neighbour.y}'] = new_wumpus_prediction[0][0][index].item()
        if self.print_output:
            print("The probabilities of the agent's neighbours being safe from a pit are: ", p_neighbours_pits)
            print("The probabilities of a wumpus being in a neighbouring cell is: ", p_neighbours_wumpus)
        # Only keep the neighbours that have a probability of being safe greater than 0.5
        safe_neighbours_pits = {key: value for key, value in p_neighbours_pits.items() if value > 0.5}
        # if the wumpus is killed, the model will have NaN values - replace them with 0
        for key, value in p_neighbours_wumpus.items():
            if isinstance(value, float) and math.isnan(value):
                p_neighbours_wumpus[key] = 0
        safe_neighbours_wumpus = {key: value for key, value in p_neighbours_wumpus.items() if value < 0.5}
        
        # can only go to a cell if the pit prob is more than 0.5 and the wumpus prob is less than 0.5
        cells_safe_from_pits = [(int(key[-2]), int(key[-1])) for key in safe_neighbours_pits.keys()]
        cells_safe_from_wummpus = [(int(key[-2]), int(key[-1])) for key in safe_neighbours_wumpus.keys()]
        
        safe_neighbours = [x for x in cells_safe_from_pits if x in cells_safe_from_wummpus] # used for building the network of the agent's locations and its neighbours
        # print("Possible safe moves are: ", safe_neighbours)

        return safe_neighbours, p_neighbours_pits, p_neighbours_wumpus
        

    def get_quickest_path(self, possible_safe_moves, safe_locations):

        # add agent's current location for the networkx graph
        possible_safe_moves.append((self.env.agent_location.x, self.env.agent_location.y)) 
        
        # first only consider cells the agent hasn't yet visited
        safe_neighbours_not_visited = []
        for safe_neighbour in possible_safe_moves:
            if safe_neighbour not in safe_locations:
                safe_neighbours_not_visited.append(safe_neighbour)
        if self.print_output:
            print("Safe neighbours that have not been visited yet are: ", safe_neighbours_not_visited)

        # Use the move planning agent to find the quickest path to the safest location
        movePlanningAgent = MovePlanningAgent(print_output=False) 
        movePlanningAgent.safe_locations = possible_safe_moves
        # build networkx graph of safe states
        G = movePlanningAgent.graph_safe_states(possible_safe_moves)

        possible_safe_moves.remove((self.env.agent_location.x, self.env.agent_location.y)) # remove the agent's current location from the list of possible safe moves
        
        if len(safe_neighbours_not_visited) == 1:
            if self.print_output:
                print("Only one safe neighbour has not been visited yet")
            fastest_safe_move_location = safe_neighbours_not_visited[0]
            fastest_safe_move_path = movePlanningAgent.find_shortest_path(G, f"({self.env.agent_location.x}, {self.env.agent_location.y}) {self.env.agent_orientation}", f"({fastest_safe_move_location[0]}, {fastest_safe_move_location[1]}) {self.env.agent_orientation}")[1]
        elif len(safe_neighbours_not_visited) > 1 or len(safe_neighbours_not_visited) == 0:
            if len(safe_neighbours_not_visited) > 1:
                if self.print_output:
                    print("More than one safe neighbour has not been visited yet")
                possible_safe_moves = safe_neighbours_not_visited
            if len(safe_neighbours_not_visited) == 0:
                if self.print_output:
                    print("all safe neighbours has already been visited")
            
            # which location takes the least number of moves to get to
            fastest_safe_move = 1000
            for loc in possible_safe_moves:
                moves = movePlanningAgent.find_shortest_path(G, f"({self.env.agent_location.x}, {self.env.agent_location.y}) {self.env.agent_orientation}", f"({loc[0]}, {loc[1]}) {self.env.agent_orientation}")[1]
                if len(moves) < fastest_safe_move:
                    fastest_safe_move = len(moves)
                    fastest_safe_move_location = loc
                    fastest_safe_move_path = moves
            
        return fastest_safe_move_location, fastest_safe_move_path






