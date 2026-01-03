# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 09:56:12 2026

@author: zjs97
"""
#导入困
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#导入数据
stock_600516 = ak.stock_zh_a_hist(symbol="600516", period="daily", start_date="20230101", end_date='20260101', adjust="qfq")
stock_000789 = ak.stock_zh_a_hist(symbol="000789", period="daily", start_date="20230101", end_date='20260101', adjust="qfq")
#数据整合
tickers={'600516':'方大炭素','000789':'万年青'}
stock_600516['日期']=pd.to_datetime(stock_600516['日期'])
stock_000789['日期']=pd.to_datetime(stock_000789['日期'])
Portfolio= pd.merge( stock_600516[['日期','收盘']].rename(columns={'收盘':'600516'}),
                   stock_000789[['日期','收盘']].rename(columns={'收盘':'000789'}),
                   on="日期",how='outer').set_index('日期')
#计算单只股票的收益率与风险
Returns=np.log(Portfolio/Portfolio.shift(1))
Annuals=np.exp(Returns.mean()*250)-1
print(Annuals)
Stds=np.sqrt(Returns.var()*250)
print(Stds)
#计算这个组合的收益与风险
def gen_weights(n):
    w=np.random.rand(n)
    return w/np.sum(w)
n=len(list(tickers))
w=gen_weights(n)
        
def Portfolio_return(w):
    return np.sum(w*Annuals)

def Portfolio_std(w):
    return np.sqrt(w.T.dot((Returns.cov()*250).dot(w)))
def gen_portfolio(times):
    for _ in range(times):
        w=gen_weights(n)
        yield(Portfolio_std(w),Portfolio_return(w),w)
#进行数据可视化        
df=pd.DataFrame(gen_portfolio(3000),columns=['标准差','收益','比重'])
matplotlib.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus'] = False  
df['Sharpe_ratio']=(df['收益']-0.019)/df['标准差']
df.plot.scatter('标准差','收益',c='Sharpe_ratio',cmap='cool',sharex=False)
plt.show()
#显示最小风险情况下的持仓比
min_vol_portfolio = df.loc[df['标准差'].idxmin()]
print(f"标准差（风险）: {min_vol_portfolio['标准差']:.4f}")
print(f"预期年化收益: {min_vol_portfolio['收益']:.4f}")
print(f"夏普比率: {min_vol_portfolio['Sharpe_ratio']:.4f}")
print("各股票权重:")
for i, (code, name) in enumerate(tickers.items()):
    print(f"  {name}({code}): {min_vol_portfolio['比重'][i]:.4f}")
    
