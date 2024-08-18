from Location import Location
from typing import List
from Orientation import Orientation
import random
from Percept import Percept
from Action import Action
WORLD_SIZE_x = 4
WORLD_SIZE_y = 4

class Environment:
    wumpus_location: Location
    wumpus_alive: bool
    agent_location: Location
    agent_orientation: Orientation
    agent_has_arrow: bool
    agent_has_gold: bool
    game_over: bool
    gold_location: Location
    pit_locations: List[Location]
    time_step: int
        
    def __init__(self, width: int, height: int, allow_climb_without_gold: bool, pit_prob: float):
        # initialize the environment state variables (use make functions below)
        self.width = width
        self.height = height
        self.allow_climb_without_gold = allow_climb_without_gold
        self.pit_prob = pit_prob

        self.wumpus_location, self.wumpus_alive = self.make_wumpus()
        self.agent_location = Location(1,1)
        self.gold_location = self.make_gold()
        self.pit_locations = self.make_pits(self.pit_prob)
        self.agent_orientation = Orientation['E']
        self.agent_has_arrow = True
        self.agent_has_gold = False
        self.game_over = False
        self.time_step = 0
        
        
    def make_wumpus(self): 
        # choose a random location for the wumpus (not bottom left corner) and set it to alive
        while True:
            location = Location.random()
            if location != Location(1,1):  # Ensure it's not the bottom left corner
                self.wumpus_location = location
                self.wumpus_alive = True
                break
        return self.wumpus_location, self.wumpus_alive

    def make_gold(self):
        # choose a random location for the gold (not bottom left corner)
        while True:
            location = Location.random()
            if location != Location(1,1):  # Ensure it's not the bottom left corner
                self.gold_location = location
                break
        return self.gold_location
        
    def make_pits(self, pit_prob: float):
        # create pits with prob pit_prob for all locations except the bottom left corner
        self.pit_locations = []
        for x in range(1, self.width+1):
            for y in range(1, self.height+1):
                if (x, y) != (1, 1):  # Exclude bottom left corner
                    if random.random() < pit_prob:
                        self.pit_locations.append(Location(x, y))
        return self.pit_locations
        
    def is_pit_at(self, location: Location) -> bool:
        # return true if there is a pit at location
        return location in self.pit_locations
        
    def is_pit_adjacent_to_agent(self) -> bool:
        # return true if there is a pit above, below, left or right of agent's current location
        for pit in self.pit_locations:
            if pit in self.agent_location.neighbours():
                return True
        return False
                
    def is_wumpus_adjacent_to_agent(self) -> bool:
        # return true if there is a wumpus adjacent to the agent
        if self.wumpus_location in self.agent_location.neighbours():
                return True
        return False
        
    def is_agent_at_hazard(self)->bool:
        # return true if the agent is at the location of a pit or the wumpus
        return (self.agent_location == self.wumpus_location and self.wumpus_alive==True) or self.is_pit_at(self.agent_location)
        
    def is_wumpus_at(self, location: Location) -> bool:
        # return true if there is a wumpus at the given location
        return self.wumpus_location == location
        
    def is_agent_at(self, location: Location) -> bool:
        # return true if the agent is at the given location
        return self.agent_location == location
        
    def is_gold_at(self, location: Location) -> bool:
        # return true if the gold is at the given location
        return self.gold_location == location
        
    def is_glitter(self) -> bool:
        # return true if the agent is where the gold is
        return self.agent_location == self.gold_location
        
    def is_breeze(self) -> bool:
        # return true if one or pits are adjacent to the agent or the agent is in a room with a pit
        if self.is_pit_adjacent_to_agent():
            return True
        if self.is_pit_at(self.agent_location):
            return True
        return False
    
    def is_stench(self) -> bool:
        # return true if the wumpus is adjacent to the agent or the agent is in the room with the wumpus
        if self.is_wumpus_adjacent_to_agent():
            return True
        if self.is_wumpus_at(self.agent_location):
            return True
        return False

    def wumpus_in_line_of_fire(self) -> bool:
        # return true if the wumpus is a cell the arrow would pass through if fired
        if self.agent_orientation == Orientation['E']:
            if (self.agent_location.x < self.wumpus_location.x) and (self.agent_location.y == self.wumpus_location.y):
                return True
        if self.agent_orientation == Orientation['W']:
            if (self.agent_location.x > self.wumpus_location.x) and (self.agent_location.y == self.wumpus_location.y):
                return True
        if self.agent_orientation == Orientation['N']:
            if (self.agent_location.x == self.wumpus_location.x) and (self.agent_location.y < self.wumpus_location.y):
                return True
        if self.agent_orientation == Orientation['S']:
            if (self.agent_location.x == self.wumpus_location.x) and (self.agent_location.y > self.wumpus_location.y):
                return True
        else:
            return False


    def kill_attempt(self) -> bool:
        # return true if the wumpus is alive and in the line of fire
        # if so set the wumpus to dead
        if self.wumpus_in_line_of_fire() and self.wumpus_alive and self.agent_has_arrow:
            self.wumpus_alive = False
        self.agent_has_arrow = False
        return self.wumpus_alive
        
    def step(self, action: Action) -> Percept:
        # for each of the actions, make any agent state changes that result and return a percept including the reward
        # First, update agent state based on action
        if action == Action.LEFT:
            # Turn agent left
            self.agent_orientation = self.agent_orientation.turn_left()
            return Percept(time_step=self.time_step+1, bump=False, breeze=self.is_breeze(), stench=self.is_stench(), scream=False, glitter=self.is_glitter(), reward=-1, done=self.game_over)

        elif action == Action.RIGHT:
            # Turn agent right
            self.agent_orientation = self.agent_orientation.turn_right()
            return Percept(time_step=self.time_step+1, bump=False, breeze=self.is_breeze(), stench=self.is_stench(), scream=False, glitter=self.is_glitter(), reward=-1, done=self.game_over)
    
        elif action == Action.FORWARD:
            # Move agent forward
            bump = self.agent_location.forward(self.agent_orientation)
            reward = -1
            if self.is_agent_at_hazard():
                reward -= 1000
                self.game_over = True
            if self.agent_has_gold == True:
                self.gold_location = self.agent_location
            return Percept(time_step=self.time_step+1, bump=bump, breeze=self.is_breeze(), stench=self.is_stench(), scream=False, glitter=self.is_glitter(), reward=reward, done=self.game_over)

        elif action == Action.GRAB:
            # Grab gold if present
            if self.is_glitter():
                self.agent_has_gold = True
            return Percept(time_step=self.time_step+1, bump=False, breeze=self.is_breeze(), stench=self.is_stench(), scream=False, glitter=self.is_glitter(), reward=-1, done=self.game_over)

        elif action == Action.SHOOT:
            # Shoot arrow 
            wumpus_orig_alive = self.wumpus_alive
            agent_has_arrow_orig = self.agent_has_arrow
            self.kill_attempt()
            wumpus_curr_alive = self.wumpus_alive
            agent_has_arrow_curr = self.agent_has_arrow
            self.agent_has_arrow = False
            if wumpus_orig_alive==True and wumpus_curr_alive == False:
                scream = True 
            else:
                scream = False
                
            if agent_has_arrow_orig == True and agent_has_arrow_curr == False:
                reward = -10
            else:
                reward = -1
            return Percept(time_step=self.time_step+1, bump=False, breeze=self.is_breeze(), stench=self.is_stench(), scream=scream, glitter=self.is_glitter(), reward=reward, done=self.game_over)

        elif action == Action.CLIMB:
            # Climb out of cave 
            if self.agent_location == Location(1,1) and self.allow_climb_without_gold==True and self.agent_has_gold==True:
                self.game_over = True
                reward = 999
            elif self.agent_location == Location(1,1) and self.allow_climb_without_gold==True and self.agent_has_gold==False:
                self.game_over = True
                reward = -1
            elif self.agent_location == Location(1,1) and self.allow_climb_without_gold==False and self.agent_has_gold==True:
                self.game_over = True
                reward = 999
            return Percept(time_step=self.time_step+1, bump=False, breeze=self.is_breeze(), stench=self.is_stench(), scream=False, glitter=self.is_glitter(), reward=reward, done=self.game_over)

        
    # Visualize the game state
    def visualize(self):
        for y in range(WORLD_SIZE_y, 0, -1):
            line = '|'
            for x in range(1, WORLD_SIZE_x+1):
                loc = Location(x, y)
                cell_symbols = [' ', ' ', ' ', ' ']
                if self.is_agent_at(loc): cell_symbols[0] = 'A'
                if self.is_pit_at(loc): cell_symbols[1] = 'P'
                if self.is_wumpus_at(loc):
                    if self.wumpus_alive:
                        cell_symbols[2] = 'W'
                    else:
                        cell_symbols[2] = 'w'
                if self.is_gold_at(loc): cell_symbols[3] = 'G'
                for char in cell_symbols: line += char
                line += '|'
            print(line)