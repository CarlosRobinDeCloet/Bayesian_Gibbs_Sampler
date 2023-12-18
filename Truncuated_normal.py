# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:42:18 2022

@author: Carlos de Cloet
"""


import numpy as np
import matplotlib.pyplot as  plt
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from statsmodels.graphics.tsaplots import plot_acf

# initializing simulation

n_sim = 10000         # number of simulations 
n_burn = 50;          # number of burn-in simulations 
nod = 2               # thinner
beta = [0, 1.5, 0, 0] # initializing beta
sigma = 0             # initializing sigma
np.random.seed(84213) # initializing random seed

drawpar = np.zeros((nod*n_sim+n_burn,5))     # matrix to store Gibbs draws

# read data
sales = pd.read_excel(r'C:\Users\Carlos de Cloet\Desktop\Bayesian Econometrics\data\sales.xls')
sales_np = np.array(sales.loc[:,"brand10"])

constant = np.ones(len(sales))

display = pd.read_excel(r'C:\Users\Carlos de Cloet\Desktop\Bayesian Econometrics\data\displ.xls')
display_np = np.array(display.loc[:,"brand10"],float)

coupon = pd.read_excel(r'C:\Users\Carlos de Cloet\Desktop\Bayesian Econometrics\data\coupon.xls')
coupon_np = np.array(coupon.loc[:,"brand10"],float)

price = pd.read_excel(r'C:\Users\Carlos de Cloet\Desktop\Bayesian Econometrics\data\price.xls')
price_np = np.array(price.loc[:,"brand10"])

X = np.transpose(np.asmatrix((constant,display_np,coupon_np,np.log(price_np))))
y = np.log(sales_np)


# Gibbs draws
for i in range(nod*n_sim + n_burn): #n_sim+n_burn
    
    if i%1000==0: print(i)
    
    # simulates sigma squared.     
    k = np.random.standard_normal(len(y))
    z = np.dot(k,k)
    resid = y - np.dot(X,beta)
    sigma = ((np.dot(resid,np.transpose(resid)))/z).item()
 

    # simulates beta_0, beta_2, beta_3
    y1 = y - beta[1]*display_np
    X1 = np.transpose(np.asmatrix((constant, coupon_np,np.log(price_np))))
    
    b_est = np.matmul(np.linalg.inv(np.matmul(np.transpose(X1),X1)),np.transpose(np.dot(np.transpose(X1),y1)))
    cov = sigma*np.linalg.inv(np.matmul(np.transpose(X1),X1))
    chol_decomp = np.linalg.cholesky(cov)
    
    w = np.array([norm.rvs(), norm.rvs(), norm.rvs()])
    w = w.reshape(3,1)
    beta_tmp = b_est - np.dot(chol_decomp,w)
    
    beta[0] = beta_tmp[0].item()
    beta[2] = beta_tmp[1].item()
    beta[3] = beta_tmp[2].item()
    
    
    # simulates beta_1
    y2 = y - beta[0]*constant- beta[2]*coupon_np - beta[3]*np.log(price_np)
    x = display_np
    
    denom = np.dot(x,y2)
    num=np.sum(x**2)
    
    bhat = denom/num
    var = sigma/num
    
    lb = norm.cdf((1-bhat)/np.sqrt(var))
    ub = 1
    beta[1] = bhat+np.sqrt(var)*norm.ppf(lb+(ub-lb)*uniform.rvs())

    # stores values
    drawpar[i,0] = beta[0]
    drawpar[i,1] = beta[1]
    drawpar[i,2] = beta[2]
    drawpar[i,3] = beta[3]
    drawpar[i,4] = sigma
        
# plots 
drawpar = drawpar[n_burn:]  # remove burn-in draws
drawpar = drawpar[range(0,n_sim*nod,nod)]

fig, (ax1, ax2) = plt.subplots(2, 1)
fig, (ax3, ax4) = plt.subplots(2, 1)
fig, (ax5) = plt.subplots(1)
ax1.set_title('Trace plot of beta_0 draws after burn-in period')
ax2.set_title('Trace plot of beta_1 draws after burn-in period')
ax3.set_title('Trace plot of beta_2 draws after burn-in period')
ax4.set_title('Trace plot of beta_3 draws after burn-in period')
ax5.set_title('Trace plot of sigma**2 draws after burn-in period')
ax1.plot(drawpar[:,0])
ax2.plot(drawpar[:,1])
ax3.plot(drawpar[:,2])
ax4.plot(drawpar[:,3])
ax5.plot(drawpar[:,4])
fig.tight_layout()
plt.show()
 

#plot_acf(drawpar[:,0],lags=10)
#plt.show()


def gewtest(y):
    draws=y+0
    nrows=len(draws) 
    mean1=np.mean(draws[0:round(0.1*nrows)],axis=0)
    mean2=np.mean(draws[round(0.5*nrows):],axis=0)
    var1=np.var(draws[0:round(0.1*nrows)],axis=0)/round(0.1*nrows)
    var2=np.var(draws[round(0.5*nrows):],axis=0)/round(0.5*nrows)
    test=(mean1-mean2)/np.sqrt(var1+var2)
    pval=norm.cdf(-abs(test))
    return(np.column_stack((mean1,mean2,np.sqrt(var1+var2),test,pval)))

def chpdi(y,p):  
    draws=y+0
    nrows=len(draws) 
    ncols=len(draws[0])
    nod = round((1-p)*nrows);   # number of draws outside interval
    if ((p<1) and (p>0) and (nod < nrows-2)):
        draws=np.sort(draws,axis=0)
        lb = draws[1:nod]                    # determine potential lower bounds
        ub = draws[nrows-nod+1:nrows]        # determine potential upper` bounds
        isize = ub-lb      # compute interval sizes
        isize = np.argmin(isize,axis=0)   # choose minimum interval   
        hpdi=np.zeros((ncols,2))
        for i in range(ncols):
            hpdi[i,0]= lb[isize[i],i]
            hpdi[i,1]= ub[isize[i],i]
    else:
       hpdi=np.zeros((ncols,2))
       hpdi[:]=np.nan 
       
    return(hpdi)