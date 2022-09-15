#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:15:02 2022

@author: pejmanjouzdani
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linalg as lg

class BasicOptimizer:
    def __init__(self, cov_matrix, rate_vec, expected_rate):
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
        self.expected_rate =expected_rate
        ##### Kernel . x = c
        ## Kernel
        self.Kernel = np.concatenate(
            (            
            cov_matrix, 
            rate_vec, 
            np.ones(rate_vec.shape)
                        ),
            axis=1)
        self.Kernel = np.concatenate(
            (self.Kernel,
             np.concatenate( (rate_vec.T, [[0, 0]]), axis=1),
             np.concatenate( (np.ones(rate_vec.shape).T, [[0, 0]]), axis=1)
             ),
            axis=0)
         
        ## c
        self.c =np.concatenate( (np.zeros(rate_vec.shape), 
                             [[expected_rate], [1]]), axis=0
                             )
        
        ##### set ws, lambda1, lambda2
        self.getW()
        
         
        
        
        
    def getW(self):
        A_inv = lg.inv(self.Kernel)
        x_calculated = A_inv.dot(self.c)
        self.ws = x_calculated[:-2]
        self.lambda1 = x_calculated[-2]
        self.lambda2 = x_calculated[-1]
        
        return self.ws, self.lambda1, self.lambda2


if __name__=='__main__':
    
    seed=10029
    np.random.seed(seed)
    
    # params
    N = 100
    num_itr= N**2
    #
    cov_matrix = np.matrix(np.random.uniform(0, 1, size=[N,N]))
    cov_matrix = (cov_matrix +cov_matrix.T)/2    
    #
    rate_vec = np.matrix(np.random.uniform(-1, 1, size=[N,1]))
    #
    expected_rate  = 0
    mpt = BasicOptimizer(cov_matrix, rate_vec, expected_rate)
    print(mpt.ws)
    print(mpt.ws.sum())
    print()
    print(mpt.lambda1)
    print()
    print(mpt.lambda2)