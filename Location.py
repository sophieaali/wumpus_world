from Orientation import Orientation
from typing import List
import random

WORLD_SIZE_x = 4
WORLD_SIZE_y = 4


class Location:
    x: int
    y: int
        
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        
    def __str__(self):
        return f'({self.x}, {self.y})'
    
    def __repr__(self) -> str:
        return f'({self.x}, {self.y})'
    
    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def is_left_of(self, location: 'Location')->bool:
    # return True if self is just left of given location
        return self.x == location.x - 1 and self.y == location.y
   
    def is_right_of(self, location: 'Location')->bool:
    # return True if self is just right of given location  
        return self.x == location.x + 1 and self.y == location.y
    
    def is_above(self, location: 'Location')->bool:
    # return True if self is immediately above given location    
        return self.x == location.x and self.y == location.y+1
    
    def is_below(self, location: 'Location')->bool:
    # return True if self is immediately below given location   
        return self.x == location.x and self.y+1 == location.y
    
    def neighbours(self)->List['Location']:
    # return list of neighbour locations    
        neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = self.x + dx, self.y + dy
            if (1 <= x <= WORLD_SIZE_x) and (1 <= y <= WORLD_SIZE_y):
                neighbours.append(Location(x, y))
        return neighbours
    
    # return True if location given is self's location
    def is_location(self, location: 'Location')->bool:
        return self.x == location.x and self.y == location.y

    def at_left_edge(self) -> bool:
    # return True if at the left edge of the grid   
        return self.x == 1
    
    def at_right_edge(self) -> bool:
    # return True if at the right edge of the grid  
        return self.x == WORLD_SIZE_x
    
    def at_top_edge(self) -> bool:
    # return True if at the top edge of the grid    
        return self.y == WORLD_SIZE_y
   
    def at_bottom_edge(self) -> bool:
     # return True if at the bottom edge of the grid    
        return self.y == 1
    
    def forward(self, orientation) -> bool:
    # modify self.x and self.y to reflect a forward move and return True if bumped a wall  
        if orientation == Orientation.N:
            if self.at_top_edge():
                return True
            else:
                self.y = self.y+1
                return False
        elif orientation == Orientation.S:
            if self.at_bottom_edge():
                return True
            else:
                self.y = self.y-1
                return False
        elif orientation == Orientation.E:
            if self.at_right_edge():
                return True
            else:
                self.x = self.x+1
                return False
        elif orientation == Orientation.W:
            if self.at_left_edge():
                return True
            else:
                self.x = self.x-1
                return False

    def set_to(self, location: 'Location'):
    # set self.x and self.y to the given location  
        self.x = location.x
        self.y = location.y
    
    @staticmethod
    def from_linear(n: int) -> 'Location':
    # convert an index from 0 to 15 to a location
        x = (n % WORLD_SIZE_x) + 1
        y = (n // WORLD_SIZE_y) + 1
        return Location (x, y)
    
    
    def to_linear(self)->int:
    # convert self to an index from 0 to 15  
        return (self.y - 1) * WORLD_SIZE_y + (self.x - 1)  
    
    
    @staticmethod
    def random() -> 'Location':
        # return a random location   

        x = random.randint(1, WORLD_SIZE_x)
        y = random.randint(1, WORLD_SIZE_y)

        while x == 1 and y == 1:
            x = random.randint(1, WORLD_SIZE_x)
            y = random.randint(1, WORLD_SIZE_y) 

        return Location(x, y)
    
    def next_cell(self, agent_orientation):
        if agent_orientation == Orientation.N:
            return Location(self.x, self.y + 1)
        elif agent_orientation == Orientation.E:
            return Location(self.x + 1, self.y)
        elif agent_orientation == Orientation.S:
            return Location(self.x, self.y - 1)
        elif agent_orientation == Orientation.W:
            return Location(self.x - 1, self.y)  
        else:
            raise ValueError("Invalid orientation") 