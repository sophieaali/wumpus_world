# wumpus_world
Wumpus world environment simulator. Completed as part of the Intelligent Agents Course at UofT

The point of WumpusWorld is to build an environment that an agent must navigate, avoiding pits and the Wumpus in order to successfully retreive the gold and get back to cell (1,1) to climb out of WumpusWorld. If an agent goes into a cell with a pit or wumpus, the agent dies. I have built 3 different agents, as described below: 

The Naive agent (Assignment 1) chooses randomly between 6 different actions: Forward, TurnLeft, TurnRight, Shoot, Grab and Climb with uniform probability. The agent acts randomly until it dies or until it finds the gold and climbs out of Wumpus World.


The MovePlanningAgent (Assignment 2) uses the Python NetworkX library to update the graph of safe locations each time the agent enters a cell it hasn't previously visited. It uses the A* algorithm and Manhattan distance as the heuristic to find the shortest path back to cell (1,1).


The ProbAgent (Assignment 3) searches the grid for the gold as safely as it can using Bayesian modelling through the Python pomegranate library. The agent makes inferences about which squares definitely don't have a pit or Wumpus, or for those that might, what the probability is so it can take calculated risks. Once the agent has the gold, it then uses the MovePlanning Agent's abilities to take the safest route back to cell (1,1) and climb out. 
