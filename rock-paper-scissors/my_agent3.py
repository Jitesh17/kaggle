import random

import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

action_list = []
observation_list = []
result_list = []


params_xgb = {
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "merror",
    "max_depth": 5,
    "eta": 0.08,
    "tree_method": "exact"
}
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
    # if observation.step == 1:
    if observation.step == 1:
        observation_list.append(observation.lastOpponentAction)
        result_list.append(
            i_win(action_list[-1], observation.lastOpponentAction))
        action = random.randint(0, 2)
        action_list.append(action)
        return action
    # if observation.step <20:
    #     observation_list.append(observation.lastOpponentAction)
    #     result_list.append(
    #         result_list[-1]+i_win(action_list[-1], observation.lastOpponentAction))
    #     action = random.randint(0, 2)
    #     action_list.append(action)
    #     return action
    observation_list.append(observation.lastOpponentAction)
    result_list.append(result_list[-1]+i_win(action_list[-1], observation.lastOpponentAction))
    
    if observation.step < 20:
        start_from = 0
    else:
        start_from = -1*random.randint(16, 20)
    X_train = np.vstack([action_list[start_from:-1],
                        observation_list[start_from:-1], 
                        result_list[start_from:-1]
                        ]).T
    y_train = np.roll(observation_list[start_from:-1], -1).T

    d_train = xgb.DMatrix(X_train, label=y_train)
    
    model = xgb.train(params=params_xgb,
                        dtrain=d_train,
                        num_boost_round=20,
                        verbose_eval=0,
                        evals=[(d_train, "train")])
    pred_train = model.predict(d_train, ntree_limit=model.best_ntree_limit)
    score = accuracy_score(pred_train, y_train)
    thresh = 0.4
    if score > thresh:
            
            
        last_data = np.array(
            # [action_list[-1], observation_list[-1]])
            [action_list[-1], observation_list[-1], result_list[-1]]).reshape(1, -1)
        # X_test = np.array([[my_actions[-1], observation.lastOpponentAction]])
        d_test = xgb.DMatrix(last_data)
        pred_obs = model.predict(d_test, ntree_limit=model.best_ntree_limit)
        action = int((pred_obs + 1) % 3)
    else:
        action = random.randint(0, 2)
    # model = XGBClassifier(
    #     learning_rate=0.01,
    #     n_estimators=30,
    #     nthread=4,
    #     use_label_encoder=False)
    # model.fit(X_train, y_train)
    # expected_observation = model.predict(last_data.reshape(1, -1))
    
    # if sum(result_list) < -3:
    # if result_list[-1] < -3:
    #     if random.randint(0, 1):
    #         action = int((expected_observation - 1) % 3)
    #     else:
    #         action = expected_observation
    # else:
    #     action = int((expected_observation + 1) % 3)
    # action = int((expected_observation + 1) % 3)
        
    action_list.append(action)
    return action
