import numpy as np
from scipy.stats import norm

# S = Stock Price
# K = Strike Price
# T = Time To Maturity
# r = risk free interest rate
# sigma = volatility of the stock 

def black_scholes_call(S,K,T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_option = S* norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return call_option

def black_scholes_put(S,K,T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_option = K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return put_option

S = 105
K = 107
T = 3/12
r = 0.04 #4 percent
sigma  = 0.10 #10 percent => Can use GARCH Model & then get your volatility from thta and put it here
#Call option price is much higher than the put option price since the stock price is greater than the strike price, alr in the money for calls

print("Call Option Price: ", black_scholes_call(S,K,T,r,sigma))
print("Put Option Price: ", black_scholes_put(S,K, T, r, sigma))

#How to model the implied volatility, can just do substitution, since premium is alr given, you can just solve for volatility, Schotastic Volatility models, 
#Vola -> finds volatility  surface, create volatility surface book, volatility is actually key thing, calculating volatility
