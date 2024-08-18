from enum import Enum, auto

class Orientation(Enum):
    E = 0
    S = 1
    W = 2
    N = 3

    def __str__(self) -> str:
        # code for function to return the letter code ("E", "S", etc.) of this instance of an orientation
        # You could create a __str__(self) for this instead of the symbol function if you prefer
        return self.name
    
    def turn_right(self) -> 'Orientation':
        # return a new orientation turned right
        # Note: the quotes around the type Orientation are because of a quirk in Python.  You can't refer
        # to Orientation without quotes until it is defined (and we are in the middle of defining it)
        if self == Orientation.E:
            return Orientation.S
        elif self == Orientation.S:
            return Orientation.W
        elif self == Orientation.W:
            return Orientation.N
        elif self == Orientation.N:
            return Orientation.E
        
    def turn_left(self) -> 'Orientation':
        # return a new orientation turned left
        if self == Orientation.E:
            return Orientation.N
        elif self == Orientation.N:
            return Orientation.W
        elif self == Orientation.W:
            return Orientation.S
        elif self == Orientation.S:
            return Orientation.E