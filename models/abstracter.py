import os,sys
import copy
import numpy as np
import pickle, joblib, time
#from interfaces import traj_stat_analysis
#from interfaces import pca_analysis
from .interfaces import grid_abs_analysis
#from interfaces import fetchCriticalState,analyze_abstraction
#from interfaces import PCA_R
from .interfaces import Grid
from multiprocessing import Process  
import scipy.stats as stats
from multiprocessing import Queue
import json

class ScoreInspector:
    
    def __init__(self, step, grid_num, state_dim, state_min, state_max):

        self.step = step
        self.grid_num = grid_num
        self.state_dim = state_dim
        self.state_min = state_min
        self.state_max = state_max
        self.basic_states = None
        self.basic_states_times = None
        self.basic_states_scores = None
        self.basic_states_proceeds = None
        #self.basic_states_values = None
        
        self.score_avg = None
        self.pcaModel = None
        self.performance_list = []
        self.avg_performance_list = []

        
        #self.QUEUE_LEN
        self.s_token = Queue(10)
        self.r_token = Queue(10)
        
        self.setup()

    
    def setup(self):

        self.min_state = np.array([self.state_min for i in range(self.state_dim)])
        self.max_state = np.array([self.state_max for i in range(self.state_dim)])
        
        self.min_avg_proceed = 0
        self.max_avg_proceed = 10

        #self.scores = scores
        self.score_avg = 0.5
        
        #self.states_info = self.setup_score_dict(states, times, proceeds, scores, values)
        self.states_info = dict()
        
        #self.pcaModel = joblib.load(config.PCA_MODEL_PATH)
        self.grid = Grid(self.min_state, self.max_state, self.grid_num)   


    def discretize_states(self, con_states):

        #pca_min, pca_max = self.pcaModel.pca_min, self.pcaModel.pca_max
        #pca_data = self.pcaModel.do_reduction(con_states)
        #abs_states = []
        #for data in pca_data:
        #    abs_state = self.grid.state_abstract(con_states = np.array([data,data]))
        #    abs_states.append(abs_state[0])
        abs_states = self.grid.state_abstract(con_states)
        return abs_states
    
    def inquery(self, pattern):
        if pattern in self.states_info.keys():
            return self.states_info[pattern]['score']
        else:
            return None

    def sync_scores(self):

        
        if self.s_token.qsize() > 0:

            new_states_info, min_avg_proceed, max_avg_proceed = self.s_token.get()

            if min_avg_proceed < self.min_avg_proceed:
                self.min_avg_proceed = min_avg_proceed
            if max_avg_proceed > self.max_avg_proceed:
                self.max_avg_proceed = max_avg_proceed
            
            self.states_info.update(new_states_info)
            self.score_avg = np.mean([self.states_info[abs_state]['score'] for abs_state in self.states_info.keys()])

            '''

            print('############################################################')
            print('Abstract states :\t', self.states_info)
            with open('state_info.json', 'w') as f:
                json.dump(self.states_info, f, indent = 6)
            print('Abstract states number :\t', len(self.states_info.keys()))
            print('Abstract traces number :\t', len(self.performance_list))
            print('Average states score :\t', self.score_avg)
            print('Queue size :\t',self.s_token.qsize())
            print('min and max proceed', self.min_avg_proceed, self.max_avg_proceed)
            print('############################################################')
            '''
    
    def start_pattern_abstract(self, con_states, rewards):
        con_states = np.array(con_states)
        t = Process(target = self.pattern_abstract, args = (con_states, rewards))
        t.daemon = True
        t.start()

    def pattern_abstract(self, con_states, rewards):

        abs_states = self.discretize_states(con_states)
        min_avg_proceed = self.min_avg_proceed
        max_avg_proceed = self.max_avg_proceed

        new_states_info = dict()
        normal_scale = self.max_avg_proceed - self.min_avg_proceed

        for i in range(len(abs_states)):
            if i + self.step >= len(abs_states):
                break
                
            proceed = sum(rewards)
            pattern = abs_states[i:i+self.step]
            pattern = '-'.join(pattern)

            if pattern in self.states_info.keys():
                self.states_info[pattern] = self.states_info[pattern]
                self.states_info[pattern]['proceed'] += proceed
                self.states_info[pattern]['time'] += 1
                average_proceed = self.states_info[pattern]['proceed'] / self.states_info[pattern]['time']
                score = (self.states_info[pattern]['proceed'] / self.states_info[pattern]['time'] - self.min_avg_proceed)  / normal_scale
                self.states_info[pattern]['score'] =  score

            else:
                new_states_info[pattern] = {}
                new_states_info[pattern]['proceed'] = proceed
                new_states_info[pattern]['time'] = 1
                average_proceed = proceed
                score = (proceed - self.min_avg_proceed) / normal_scale
                new_states_info[pattern]['score'] =  score
            
            if average_proceed < min_avg_proceed:
                min_avg_proceed = average_proceed
            if average_proceed >max_avg_proceed:
                max_avg_proceed = average_proceed

        self.s_token.put((new_states_info, min_avg_proceed, max_avg_proceed))

    


class Abstracter:
    
    def __init__(self, step=1, decay=0.1, repair_scope=0.25):
        self.con_states = []
        self.con_values = []
        self.con_reward = []
        self.con_dones  = []
        self.step = step
        self.decay = decay
        self.repair_scope = repair_scope
        self.inspector = None
        
    def append(self, con_state, reward, done):
        self.con_states.append(con_state)
        self.con_reward.append(reward)
        self.con_dones.append(done)

        if done:
            self.inspector.start_pattern_abstract(self.con_states, self.con_reward)
            self.clear()
    
    def clear(self):
        self.con_states = []
        self.con_reward = []
        self.con_dones  = []
    
    def handle_pattern(self,con_states,rewards,step, total_step):
        
        abs_pattern = self.inspector.discretize_states(con_states)
        
        if len(abs_pattern) != self.step:
            return rewards[0]
        pattern = '-'.join(abs_pattern)
            
        final_score = self.inspector.inquery(pattern)
        if final_score:
            if final_score < self.repair_scope:
                #print('original_reward:\t', rewards[0], final_score, self.inspector.score_avg)
                rewards[0] += (final_score - self.inspector.score_avg) * self.decay * (step/total_step)
                #print('new_reward:\t', rewards[0])
                

        return rewards[0]



    def reward_shaping(self, state_list, reward_list, step, total_step):
        
        shaping_reward_list = copy.deepcopy(reward_list)

        for i in range(len(state_list) - self.step):

            target_states = state_list[i:i+self.step]
            target_rewards = reward_list[i:i+self.step]

            shaped_reward = self.handle_pattern(target_states, target_rewards, step, total_step)
            shaping_reward_list[i] = shaped_reward
        
        shaping_reward_list = np.array(shaping_reward_list)
        return shaping_reward_list