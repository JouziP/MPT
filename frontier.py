#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:48:30 2022

@author: pejmanjouzdani
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from basic_optimizer import BasicOptimizer


class Frontier:
    def __init__(self, cov_matrix, rate_vec, expected_returns_domain):
        self.cov_mtx =cov_matrix
        self.rate_vec =rate_vec        
        self.expected_returns_domain =expected_returns_domain
        self.results = self.getFrontier(expected_returns_domain, 
                cov_matrix,
                rate_vec,
                )
        
    def plotFrontier(self):
        fig, ax = plt.subplots(1,1, figsize=[10,6])
        ax.scatter(self.results['portfolio_risk'],
                    self.results['portfolio_return'])  
        ax.set_xlabel('risk (vol)')
        ax.set_ylabel('return (expected rate)')
        fig.show()
        
        


    def getFrontier(self, expected_returns_domain, 
                    cov_matrix,
                    rate_vec,
                    ):
        results=[]
        for expected_rate in expected_returns_domain:
            mpt = BasicOptimizer(cov_matrix, rate_vec, expected_rate)
            mpt.getOptimizedMean()
            mpt.getOptimizedRisk()
            
            results.append(
                [expected_rate,  
                 mpt.portfolio_optimized_risk,
                mpt.portfolio_optimized_rate]+\
                mpt.ws.T.tolist()[0]+\
                [mpt.lambda1,
                mpt.lambda2,
                ]
                )
            
        results = pd.DataFrame(results)
        results.columns = ['expected_return',
                            'portfolio_risk',
                            'portfolio_return']+\
            ['w'+str(i) for i in range(len(mpt.ws))]+\
            ['lambda1', 'lambda2']
                                
        return results
    

if __name__=='__main__':    
    # ###############   Real values
    
    from datetime import datetime
    
    from utilityFunctions import getYahooMultiPrice, getRates
    from utilityFunctions import getCovMtx, getMeanVec
    
    tickers =  [
                'ASML',
                'AAPL',
                'AAL', 
                'DAL', 
                
                'GOOG',                
                'JBLU', 
                'KLAC',
                'NCLH', 
                '^GSPC',
                'KIND',
                'CSX',                
                'PHM',
                'MRNA',
                'WY',
                'PFE',
                'RKLB',
                ]    
    when = 'High'
    d1 = datetime(2022, 1, 20)
    d2 = datetime(2022, 12, 5)
    ticker='DAL'
    interval='1d'      
    rate_interval = 1
    ### MPT specific
    expected_returns_domain = np.linspace(0.001, 1,20)     
    #######################################################    
    closes = getYahooMultiPrice(tickers, d1, d2, interval, when)
    print(closes)
    print()
    
    rates = getRates(closes, rate_interval)
    print(rates)
    
    rate_vec = getMeanVec(rates)
    cov_matrix = getCovMtx(rates)

    myF = Frontier(cov_matrix, rate_vec, expected_returns_domain)
    myF.plotFrontier()