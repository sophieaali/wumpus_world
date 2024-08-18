from Action import Action
import random
from Environment import Environment
from Percept import Percept

class NaiveAgent:
    
    def choose_action(self):
        # return a randomly chosen Action
        self.nextAction = Action(random.randint(0, 5))
        return self.nextAction
    
    def run(self, WORLD_SIZE_x, WORLD_SIZE_y, allow_climb_without_gold, pit_prob):
        env = Environment(WORLD_SIZE_x, WORLD_SIZE_y, allow_climb_without_gold, pit_prob)
        cumulative_reward = 0
        percept =  Percept(time_step=0, bump=False, breeze=env.is_breeze(), stench=env.is_stench(), scream=False, glitter=env.is_glitter(), reward=0, done=env.game_over)
        while not percept.done:
            env.visualize()
            print('Percept:', percept)
            self.choose_action()
            print()
            print('Action:', self.nextAction)
            print()
            percept = env.step(self.nextAction)
            cumulative_reward += percept.reward
        env.visualize()
        print('Percept:', percept)
        print('Cumulative reward:', cumulative_reward)