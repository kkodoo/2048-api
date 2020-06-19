from game2048.game import Game
from game2048.displays import Display
from sklearn.externals import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from collections import OrderedDict
import numpy as np

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game,display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 4096
    N_TESTS = 10

    '''====================
    Use your own agent here.'''
    from game2048.agent_voting import CNN_voting as TestAgent
    '''===================='''
    #model1 = MyAgent()
    #model2 = Net()
    #model3 = MyAgent()
    #model3 = Net2()
    #model4 = Net2()
    #model3 = DQN(Env, CNN)
    #model3.resume('./game2048/model3_pretrain_params.t7')
    scores = []
    t2 = time.time()
    for i in range(N_TESTS):
        t1 = time.time()
        score = single_run(GAME_SIZE,SCORE_TO_WIN,AgentClass=TestAgent)
        t = time.time() - t1
        print("Time = ",t) 
        scores.append(score)
        print("Score = ",score)
    
    total = time.time()-t2
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
    print(scores)
    print("Total time = ",total)

    score, count = np.unique(scores, return_counts = True)
    print('Score Table :')
    print('-'*15)
    print(' {0:^5} | {1:^5} '.format('Score', 'Count'))
    print('-'*15)
    for s, c in zip(score, count):
        print(' {0:^5} | {1:^5} '.format(s,c))
    print('-'*15)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
