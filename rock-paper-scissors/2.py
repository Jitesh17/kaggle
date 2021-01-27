import random

def copy_opponent(observation, configuration):
    if observation.step > 0:
        if random.randint(0, 1):
            return (observation.lastOpponentAction + 1) % 3
        else:
            return (observation.lastOpponentAction + 2) % 3
    else:
        return random.randrange(0, configuration.signs)
