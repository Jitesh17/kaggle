# In[]
from kaggle_environments import evaluate, make, utils

from agents import (copy_opponent, counter_reactionary, paper, reactionary,
                    rock, scissors, statistical)
# from my_agent2 import Agent
from x import xgb_agent
from q_l import move
import q2
import my_agent1_1
import random

import my_agent2
import my_agent3
import time
env = make("rps", debug=True)

agents = {
    "rock": rock,
    "paper": paper,
    "scissors": scissors,
    "copy_opponent": copy_opponent,
    "reactionary": reactionary,
    "counter_reactionary": counter_reactionary,
    "statistical": statistical
}
# print(list(env.agents))
def rand(observation, configuration):
    return random.randint(0, 1)
# def rock(observation, configuration):
#     return 0
# env.run(["rock-paper-scissors/my_agent2.py", rock])
# for agent in agents:
#     print(agent)
#     env.reset()
#     # print(agents[agent])
#     # env.run([Agent, agents[agent]])
#     env.run([Agent, agents[agent]])
#     # env.run([xgb_agent, rock])

#     env.render(mode="ipython", width=500, height=400)
#     # env.
tic = time.clock()
env.reset()
# print(agents[agent])
# env.run([Agent, agents[agent]])
fighters = [my_agent3.Agent, my_agent1_1.Agent]
fighters = [my_agent3.Agent, move]
fighters = [my_agent3.Agent, move]
fighters = [q2.move, move]
fighters = [my_agent1_1.Agent, move]
fighters = [rand, move]
# fighters = [my_agent1_1.Agent, rock]
# fighters = [my_agent3.Agent,rock]
scores = []
n = 1
for i in range(n):
    scores.append( evaluate('rps', fighters, configuration={'episodeSteps': 1000})[0][0])
    # print(scores[i])
print(f'avg. score: {sum(scores)/len(scores)}')

# env.run(fighters)
# env.render(mode="ipython", width=500, height=400)
toc = time.clock()
print(f'time: {toc - tic}')
# %%
