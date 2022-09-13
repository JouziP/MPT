#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:15:02 2022

@author: pejmanjouzdani
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class BasicOptimizer:
    def __init__(self,x_init, cov_matrix, rate_vec, expected_rate , delta):
        '''
            Parameters
            ----------
            x_current : vector
                N=2 vector, x_current[:N] = ws, x_current[N:N+1] = lambda1, and
                x_current[N+1:N+2] = lambda2 
                
            cov_matrix : matrix
                N by N matrix of covarince
            rate_vec : vector
                N by 1 vector of historical rates
            expected_rate: float
                is the expected rate of the portfolio
        '''
        self.cov_mtx =cov_matrix
        self.rate_vec =rate_vec
        self.x_current =x_init
        self.expected_rate =expected_rate
        self.delta =delta
        self.ws_optimized = np.copy(x_init[:N])
        
        
    ########### Utilities
    def portfolioRate(self):
        ws = np.matrix(self.ws_optimized)
        ws = ws.reshape([len(self.ws_optimized), 1])
        rate_portfolio_optimized = ws.T.dot(self.rate_vec)[0,0]        
        return rate_portfolio_optimized

    def portfolioRisk(self):
        ws = np.matrix(self.ws_optimized)
        ws = ws.reshape([len(self.ws_optimized), 1])
        risk_portfolio_optimized = (ws.T.dot(self.cov_mtx)).dot(ws) [0,0]        
        return risk_portfolio_optimized
        
    ###########
    def optimizer(self, num_itr,
                        x_current, 
                        cov_matrix, 
                        rate_vec, 
                        expected_rate, 
                        delta):
        
        reslts=[]
        for it in range(num_itr):
     
            E_current = self.lagrangian(x_current, 
                                           cov_matrix, 
                                           rate_vec, 
                                           expected_rate, 
                                           delta)
            
            self.ws_optimized = np.copy(x_current[:N])
            
            self.rate_portfolio_optimized   =  self.portfolioRate()
            self.risk_portfolio_optimized   =  self.portfolioRisk()
            
            ####            
            reslts.append([it, E_current, 
                           self.rate_portfolio_optimized,
                           self.risk_portfolio_optimized,
                           sum(x_current[:N])])
            x_new = self.canonicalMomentum(x_current, cov_matrix, rate_vec, expected_rate , delta)            
            x_current = np.copy(x_new)
            
        reslts = pd.DataFrame(reslts, columns=['it', 'E', 'Rate', 'Risk' , 'norm'])
        
        
        
        
        return reslts
    
    
            
    def lagrangian(self, x_current, cov_matrix, rate_vec, expected_rate , delta):
        '''
        
        computes a lagrangian L = L(ws,  lambda1). Our goal is to minimize 
        this objective function
    
        Parameters
        ----------
        x_current : vector
            N=2 vector, x_current[:N] = ws, x_current[N:N+1] = lambda1, and
            x_current[N+1:N+2] = lambda2 
            
        cov_matrix : matrix
            N by N matrix of covarince
        rate_vec : vector
            N by 1 vector of historical rates
        expected_rate: float
            is the expected rate of the portfolio
    
        Returns
        -------
        None.
    
        '''
        
        
        # number of weights
        N = x_current.shape[0] - 1
        
        # weights
        ws = np.matrix(x_current[:N]).reshape(N, 1)
        lambda1 = x_current[N:N+1][0]
        
        
        
        ####
        E = (ws.T.dot(cov_matrix).dot(ws))[0,0] +\
            lambda1 * ( (rate_vec.T.dot(ws))[0,0] - expected_rate)
            
        return E


    def canonicalMomentum(self,x_current, cov_matrix, rate_vec, expected_rate , delta):
        '''
        
        adjust the cordinate values by 
        
        w(t+delta) =  w(t) - delta * (\partial L/ \partial w)
        
        
        Parameters
        ----------
        x_current : vector
            N=2 vector, x_current[:N] = ws, x_current[N:N+1] = lambda1              
        cov_matrix : matrix
            N by N matrix of covarince
        rate_vec : vector
            N by 1 vector of historical rates
        expected_rate: float
            is the expected rate of the portfolio
        
    
        Returns
        -------
        x_new: updated x variables.
    
        '''
        
        
        # number of weights
        N = x_current.shape[0] - 1
        
        # weights
        ws = np.matrix(x_current[:N]).reshape(N, 1)
        lambda1 = x_current[N:N+1][0]
        
        
        
        # delta_w
        delta_w = (cov_matrix + cov_matrix.T).dot(ws) 
        delta_w += lambda1*rate_vec
        
        ws_new = np.copy(ws) - delta * delta_w
        
        ### enforce normalization
        ws_new = ws_new/sum(ws_new)[0,0]
        
        # delta_lambda1
        delta_lambda1 = (rate_vec.T.dot(ws))[0,0] - expected_rate
        lambda1 = lambda1 - delta * delta_lambda1
        
    
        
        x_new = np.array(ws_new.T.tolist()[0]+ [lambda1])
        return x_new



if __name__=='__main__':
    
    seed=10029
    np.random.seed(seed)
    
    # params
    N = 10
    num_itr= N**2
    #
    cov_matrix = np.matrix(np.random.uniform(0, 1, size=[N,N]))
    cov_matrix = (cov_matrix +cov_matrix.T)/2    
    #
    rate_vec = np.matrix(np.random.uniform(-1, 1, size=[N,1]))
    #
    expected_rate  = 1
    #
    delta = 0.01
    
    ####
    ws_current = np.random.uniform(0, 1, size=N)
    ws_current = ws_current/sum(ws_current)
    
    lambda1 = 100
    
    
    x_init = ws_current.tolist() + [lambda1]
    x_init = np.array(x_init)
    
    myBO = BasicOptimizer(x_init, cov_matrix, rate_vec, expected_rate , delta)
    reslts = myBO.optimizer(num_itr, 
                                      x_init, 
                                      cov_matrix, 
                                      rate_vec, 
                                      expected_rate, 
                                      delta)
    
    
    plt.plot(reslts['it'], np.array([reslts['Risk'], reslts['Rate']]).T, '--')
    