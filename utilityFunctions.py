#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:49:57 2022

@author: pejmanjouzdani
"""


import time
import pandas as pd
import numpy as np
from datetime import datetime

def getYahooPrice(ticker, d1, d2, interval = '1d'):
    '''
    parameters:
        ticker: str, symbol of the stock or security
        d1: datetime object, begining of the period 
        d2: datetime object, end of the period 
        interval: '1d', '1wk', '1mo'
    '''    
    period1 = int(time.mktime(datetime(d1.year, d1.month, d1.day).timetuple()))
    period2 = int(time.mktime(datetime(d2.year, d2.month, d2.day).timetuple()))
    # interval = '1mo'  ### '1d' , '1wk', '1mo'
   
    
    url0 ='https://query1.finance.yahoo.com/v7/finance/download/%s?'%(ticker)
    url1 ='period1=%s&period2=%s'%(period1, period2)
    url2 ='&interval=%s&events=history&includeAdjustedClose=true'%(interval)
    
    
    url = url0+url1 + url2
    
    df = pd.read_csv(url)   
    df = df.set_index('Date')
    return df

def getYahooMultiPrice(tickers, d1, d2, interval='1d', when='Close'):
    '''
    parameters:
        tickers: list, list of symbols (str) of the stock or securities
        d1: datetime object, begining of the period 
        d2: datetime object, end of the period 
        interval: '1d', '1wk', '1mo'
        when: 'Close', 'Open', 'Low', 'High'
    '''
    
    rslts= pd.DataFrame()
    
    for ticker in tickers:
        df = getYahooPrice(ticker, d1, d2, interval)
        rslts = pd.concat((rslts, df[[when]]), axis=1)
    rslts.columns = tickers
    
    return rslts


def getRates(prices, interval=1):
    '''
    computes the compounded rate as log(P2/P1), log is the natural log
    P2 is price at t2 and P1 price at t1, t2>t1.

    Parameters
    ----------
    prices : Pandas DataFrame
        columns = [ticker1 , ... , ], where ticker is a str.
        index= dates
    interval : Int,  optional
        default if interval = 1, the unit is based on the unit 
        of the index/dates

    Returns
    -------
    rates : Pandas DataFrame
        DESCRIPTION.

    '''
    pass
    try:
        assert(prices.shape[0]>interval)
    except:
        print('Number of prices are less than the requested interval')

    rates = pd.DataFrame(prices[interval:].values/prices[:-interval].values,
                         columns=prices.columns)
    rates.index = prices.index[interval:]
    
    rates = np.log(rates)
    
    
    return rates

def getCovMtx(rates):
    '''
    

    Parameters
    ----------
    rates : Pandas DataFrame
        has N tickers.

    Returns
    -------
    C : numpy matrix
        N by N matrix

    '''
    C = np.matrix(rates.cov().values)
    return C

def getMeanVec(rates):
    means = rates.mean(axis=0)
    means = np.matrix(means.values).T
    return means





if __name__=='__main__':  
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
    
    
    #######################################################
    print('test %s '%(getYahooPrice.__name__) )    
    df = getYahooPrice(ticker, d1, d2, interval)
    print(df)
    print()
    print()

    #######################################################
    print('test %s '%(getYahooMultiPrice.__name__) )
    closes = getYahooMultiPrice(tickers, d1, d2, interval, when)
    print(closes)
    print()
    
    rate_interval = 1
    rates = getRates(closes, rate_interval)
    print(rates)