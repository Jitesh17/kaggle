import random

import numpy as np

from xgboost.sklearn import XGBClassifier

action_list = []
observation_list = []
result_list = []


def i_win(me, you):
    return int((me - you + 4) % 3) - 1

# for i in range(3):
#     text = ""
#     for j in range(3):
#         text += f'{i_win(i, j)} '
#     print(f'{text}')


def Agent(observation, configuration):
    global action_list, observation_list, result_list
    if observation.step == 0:
        action = random.randint(0, 2)
        action_list.append(action)
        return action
    if observation.step == 1:
        observation_list.append(observation.lastOpponentAction)
        result_list.append(
            i_win(action_list[-1], observation.lastOpponentAction))
        action = random.randint(0, 2)
        action_list.append(action)
        return action
    observation_list.append(observation.lastOpponentAction)
    result_list.append(i_win(action_list[-1], observation.lastOpponentAction))
    if observation.step < 20:
        start_from = 0
    else:
        start_from = -1*random.randint(10, 20)
    X_train = np.array([action_list[start_from:-1],
                        observation_list[start_from:-1], result_list[start_from:-1]]).T
    y_train = np.roll(observation_list[start_from:-1], -1).T

    model = XGBClassifier(
        learning_rate=0.01,
        n_estimators=20,
        nthread=4)
    model.fit(X_train, y_train)
    last_data = np.array([action_list[-1], observation_list[-1], result_list[-1]])
    expected_observation = model.predict(last_data.reshape(1, -1))
    action = int((expected_observation + 1) % 3)
    action_list.append(action)
    return action