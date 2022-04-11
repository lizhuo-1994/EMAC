import os,sys
import copy
import numpy as np
import pickle, joblib, time
from interfaces import traj_stat_analysis
from interfaces import pca_analysis
from interfaces import grid_abs_analysis
from interfaces import fetchCriticalState,analyze_abstraction
from interfaces import PCA_R
from interfaces import Grid
from multiprocessing import Process  
import scipy.stats as stats


class ScoreInspector:
    
    def __init__(self, step):

        self.step = step
 
        self.basic_states = None
        self.basic_states_times = None
        self.basic_states_scores = None
        self.basic_states_proceeds = None
        #self.basic_states_values = None

        self.max = 0
        self.min = 0
        
        self.score_avg = None
        self.pcaModel = None
        self.performance_list = []
        self.avg_performance_list = []

        
        #self.QUEUE_LEN
        self.s_token = Queue(10)
        self.r_token = Queue(10)
        
        self.setup()

    
    def setup(self):

        self.max_state = None
        self.min_state = None
        
        #self.scores = scores
        self.score_avg = np.mean(scores)
        
        #self.states_info = self.setup_score_dict(states, times, proceeds, scores, values)
        self.states_info = self.setup_score_dict(states, times, proceeds, scores)
        
        #self.pcaModel = joblib.load(config.PCA_MODEL_PATH)
        self.grid = Grid(self.max_state, self.min_state, self.grid_num)   
    
    #def setup_score_dict(self, states, times, proceeds, scores, values):
    def setup_score_dict(self, states, times, proceeds, scores):
        
        d = dict()
        for i in range(len(states)):
            d[states[i]] = {}
            d[states[i]]['time'] = times[i]
            d[states[i]]['proceed'] = proceeds[i]
            d[states[i]]['score'] = scores[i]
            
        return d
    
    def discretize_states(self, con_states):

        #pca_min, pca_max = self.pcaModel.pca_min, self.pcaModel.pca_max
        
        #pca_data = self.pcaModel.do_reduction(con_states)
        #abs_states = []
        #for data in pca_data:
        #    abs_state = self.grid.state_abstract(con_states = np.array([data,data]))
        #    abs_states.append(abs_state[0])

        abs_states = []
        for data in con_states:
            abs_state = self.grid.con_states(con_states = np.array([data,data]))
            abs_states.append(abs_state[0])
        return abs_states
    
    def inquery(self, pattern):
        if pattern in self.states_info.keys():
            return self.states_info[pattern]['score']
        else:
            return None

    def sync_scores(self):

        
        if self.s_token.qsize() > 0:
            
            new_states_info, new_min, newmax = self.s_token.get()

            self.states_info.update(new_states_info)
            self.min = new_min
            self.max = new_max
            self.score_avg = np.mean([i['score'] for i in self.states_info.keys()])
            
            print('############################################################')
            print('Abstract states number :\t', len(self.states_info.keys()))
            print('Abstract traces number :\t', len(self.performance_list))
            print('Average states score :\t', self.score_avg)
            print('Queue size :\t',self.s_token.qsize())
            print('min and mx scores', self.min, self.max)
            print('############################################################')
    
    def start_pattern_abstract(self, con_states, rewards, values):

        t = Process(target = self.pattern_abstract, args = (con_states, rewards))
        t.daemon = True
        t.start()

    def pattern_abstract(self, con_states, rewards):

        abs_states = self.discretize_states(con_states)
        
        new_states_info = {}
        
        for i in range(len(abs_states)):
            if i + self.step >= len(abs_states):
                break
                
            basic_proceed = sum(rewards)
            basic_pattern = abs_states[i:i+self.step]
            basic_pattern = '-'.join(basic_pattern)

            if basic_pattern in self.states_info.keys():
                new_states_info[basic_pattern]['proceed'] += basic_proceed
                new_states_info[basic_pattern]['time'] += 1
                score = (new_states_info[basic_pattern]['proceed'] / new_states_info[basic_pattern]['time'] - self.min) / (self.max - self.min)
                new_states_info[basic_pattern]['score'] =  score

            else:
                new_states_info[basic_pattern] = {}
                new_states_info[basic_pattern]['proceed'] = basic_proceed
                new_states_info[basic_pattern]['time'] = 1
                score = (basic_proceed - self.min) / (self.max - self.min)
                new_states_info[basic_pattern]['score'] =  score
            
            if score < self.min:
                self.min = score
            if score > self.max:
                self.max = score

        self.s_token.put(new_states_info, self.min, self.max)

    


class Abstracter:
    
    def __init__(self, step, decay, repair_scope):
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
    
    def handle_pattern(self,con_states,rewards):
   
        abs_pattern = self.inspector.discretize_states(con_states)

        if len(abs_pattern) != self.step:
            return rewards[0]

        pattern = '-'.join(abs_pattern)
            
        final_score = self.inspector.inquery(pattern)
        if final_score::
            if final_score < self.repair_scope:
                #print('original_reward:\t', rewards[0], final_score, self.inspector.score_avg)
                rewards[0] += (final_score - self.inspector.score_avg) * self.decay
                #print('new_reward:\t', rewards[0])

        return rewards[0]



    def reward_shaping(self, state_list, reward_list):

        shaping_reward_list = copy.deepcopy(reward_list)

        for i in range(len(state_list) - self.step):

            target_states = state_list[i:i+self.step]
            target_rewards = reward_list[i:i+self.step]

            shaped_reward = self.handle_pattern(target_states, target_rewards)
            shaping_reward_list[i] = shaped_reward
        
        shaping_reward_list = np.array(shaping_reward_list)

        return shaping_reward_list