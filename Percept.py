class Percept():
    time_step: int
    bump: bool
    breeze: bool
    stench: bool
    scream: bool
    glitter: bool
    reward: int
    done: bool
        
    def __init__(self, time_step: int, bump: bool, breeze: bool, stench: bool, scream: bool, glitter: bool, reward: int, done: bool):
        # add code to set the instance variables of the percept
        self.time_step = time_step
        self.bump = bump
        self.breeze = breeze
        self.stench = stench
        self.scream = scream
        self.glitter = glitter
        self.reward = reward
        self.done = done

    def __str__(self):
        # add helper function to return the contents of a percept in a readable form
        return f"Percepts: Time Step={self.time_step}, Bump={self.bump}, Breeze={self.breeze}, Stench={self.stench}, Scream={self.scream}, Glitter={self.glitter}, Reward={self.reward}, Done={self.done}"