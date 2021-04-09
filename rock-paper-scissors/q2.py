import numpy as np
import random

def who_won(me,opponent):
    return int((me - opponent + 4) % 3) - 1


Q = np.zeros((9, 3))
alpha = 0.8
alpha_decay = 0.99
discount = 0.31
epsilon = 0.82
epsilon_decay = 0.99

STATES = {(0, 0): 0,
          (0, 1): 1,
          (0, 2): 2,
          (1, 0): 3,
          (1, 1): 4,
          (1, 2): 5,
          (2, 0): 6,
          (2, 1): 7,
          (2, 2): 8}

current_state = 0
current_action = 0


def move(observation):
    global current_state
    global current_action
    global STATES
    global discount
    global alpha
    global Q
    global epsilon
    global epsilon_decay
    global alpha_decay
    
    if observation.step == 0:
        current_action = int(np.random.randint(0,3))
        return current_action
    elif observation.step == 1:
        current_state = STATES[(current_action,observation.lastOpponentAction)]
        
        if epsilon > random.uniform(0,1):
            current_action = int(np.random.randint(0,3))
            return current_action
        else:
            current_action = int(Q[current_state,:].argmax())
            return current_action
        
        return current_action 
    else:
        reward = who_won(current_action,observation.lastOpponentAction)
        next_state = STATES[(current_action,observation.lastOpponentAction)]
        
        discounted_next_state = alpha*(reward+
                                       discount*Q[next_state,Q[next_state,:].argmax()] - 
                                       Q[current_state,current_action])
        
        Q[current_state,current_action] = Q[current_state,current_action] + discounted_next_state
        current_state = STATES[(current_action,observation.lastOpponentAction)]
        
        
        
        if epsilon > random.uniform(0,1):
            current_action = int(np.random.randint(0,3))
        else:
            current_action = int(Q[current_state,:].argmax())
         
        alpha*=alpha_decay
        epsilon*=epsilon_decay
        return current_action