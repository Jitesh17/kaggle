import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

my_actions = np.empty((0, 0), dtype=int)
op_actions = np.empty((0, 0), dtype=int)

params_xgb = {
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "merror",
    "max_depth": 5,
    "eta": 0.08,
    "tree_method": "exact"
}

def xgb_agent(observation, configuration):
    
    global my_actions, op_actions
    
    if observation.step == 0:
        
        my_action = random.randint(0, 2)
        my_actions = np.append(my_actions, my_action)
        return my_action
    
    if observation.step == 1:
        
        my_action = random.randint(0, 2)
        my_actions = np.append(my_actions, my_action)
        op_actions = np.append(op_actions, observation.lastOpponentAction)
            
        return my_action
    
    # 2戦目以降
    # after the 2nd time
    else:
        op_actions = np.append(op_actions, observation.lastOpponentAction)
        
        # -1までにする(つまり、最後の手を学習に使わない)のは、
        # 目的変数を次の相手の手とするから(最後の手まで学習に使うと、
        # その行の目的変数が存在しなくなってしまう)
        X_train = np.vstack([my_actions[:-1], op_actions[:-1]]).T
        
        tmp = np.roll(op_actions, -1)    # 1つずつ前にずらす。1番前のものは一番後ろに持っていく。
        y_train = tmp[:-1].T
        
        # train dataのサンプル数が21以上となった場合、計算時間短縮のため直近20回のデータのみ使う。
        if len(X_train) >= 21:
            X_train = X_train[-20:, :]
            y_train = y_train[-20:]
            
        
        d_train = xgb.DMatrix(X_train, label=y_train)
        
        model = xgb.train(params=params_xgb,
                          dtrain=d_train,
                          num_boost_round=50,
                          verbose_eval=0,
                          evals=[(d_train, "train")])
        
        pred_train = model.predict(d_train, ntree_limit=model.best_ntree_limit)
        score = accuracy_score(pred_train, y_train)
        
        # もし予測精度が1/3以上のときはモデルの予測値をもとに出す手を決定する。
        # If the accuracy of the prediction is more than 1/3, the move to be made is determined based on the prediction of the model.
        if score >= 0.33:
            
            X_test = np.array([[my_actions[-1], observation.lastOpponentAction]])
            d_test = xgb.DMatrix(X_test)
            pred = model.predict(d_test, ntree_limit=model.best_ntree_limit)
            pred = int(pred)

            # 予測した相手の手に対して勝つ手を選択
            if pred == 0:
                my_action = 1
            elif pred == 1:
                my_action = 2
            elif pred == 2:
                my_action = 0
            else:
                my_action = 0
        # もし予測精度が1/3より低いときはランダムに手を出した方がマシ。
        # If the prediction accuracy is lower than 1/3, it is better to make a random move.
        else:
            my_action = random.randint(0, 2)
            
        my_actions = np.append(my_actions, my_action)
        
        return my_action