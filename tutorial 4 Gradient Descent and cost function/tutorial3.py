import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y):
    m_curr = b_curr = 0
    rate = 0.01
    iterations=10000
    n =  len(x)
    plt.scatter(x,y,color='red',marker='+',linewidth='5')
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        plt.plot(x,y_predicted,color='blue')
        md = -(2/n)*sum(x*(y-y_predicted))
        yd = -(2/n)*sum(y-y_predicted)
        
        m_curr = m_curr - rate * md
        b_curr = b_curr -rate * yd
        
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
        
gradient_descent(x,y)
        
plt.plot(gradient_descent(x,y))