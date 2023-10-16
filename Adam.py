import numpy as np
import matplotlib.pyplot as plt
import math

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


# Gradient descent
def gd_2d(x1, x2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2)

x1, x2 = -5, -2
eta = 0.4

res = [(x1, x2)]
print(f'res before : {res}')

for i in range(20):
    x1, x2 = gd_2d(x1, x2)
    res.append((x1, x2))

print(f'epoch {i+1}, x1 {x1:.6f}, x2 {x2:.6f}')
print(f'res after : {res}')  
x1, x2 = zip(*res)
print(f'x1 : {x1}')  
print(f'x2 : {x2}')  

plt.plot(x1, x2, '-o', color='red')
x1 = np.arange(min(x1) - 1, max(x1) + 1, 0.1)
x2 = np.arange(min(x2) - 1, max(x2) + 1, 0.1)
x1, x2 = np.meshgrid(x1, x2)
plt.contour(x1, x2, f_2d(x1, x2), colors='blue')
plt.title('Gradient descent') 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 


# Momemtum
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

# eta, gamma = 0.4, 0.5   
eta, gamma = 0.05, 0.9

x1, x2 = -5, -2
v1, v2 = 0, 0

res = [(x1, x2)]
print(f'res before : {res}')

for i in range(20):
    x1, x2, v1, v2 = momentum_2d(x1, x2, v1, v2)
    res.append((x1, x2))

print(f'epoch {i+1}, x1 {x1:.6f}, x2 {x2:.6f}')
print(f'res after : {res}')  
x1, x2 = zip(*res)
print(f'x1 : {x1}')  
print(f'x2 : {x2}')  

plt.plot(x1, x2, '-o', color='red')
x1 = np.arange(min(x1) - 1, max(x1) + 1, 0.1)
x2 = np.arange(min(x2) - 1, max(x2) + 1, 0.1)
x1, x2 = np.meshgrid(x1, x2)
plt.contour(x1, x2, f_2d(x1, x2), colors='blue')
plt.title('Momemtum') 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()



# Adagrad
def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2    

eta = 0.4

x1, x2 = -5, -2
s1, s2 = 0, 0

res = [(x1, x2)]
print(f'res before : {res}')

for i in range(20):
    x1, x2, s1, s2 = adagrad_2d(x1, x2, s1, s2)
    res.append((x1, x2))

print(f'epoch {i+1}, x1 {x1:.6f}, x2 {x2:.6f}')
print(f'res after : {res}')  
x1, x2 = zip(*res)
print(f'x1 : {x1}')  
print(f'x2 : {x2}')  

plt.plot(x1, x2, '-o', color='red')
x1 = np.arange(min(x1) - 1, max(x1) + 1, 0.1)
x2 = np.arange(min(x2) - 1, max(x2) + 1, 0.1)
x1, x2 = np.meshgrid(x1, x2)
plt.contour(x1, x2, f_2d(x1, x2), colors='blue')
plt.title('Adagrad') 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 



# Adam
def adam_2d(x1, x2, s1, s2 , t):
    beta1, beta2, eps = 0.9, 0.99, 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1[0] = beta1 * s1[0] + (1 - beta1) * g1
    s2[0] = beta1 * s2[0] + (1 - beta1) * g2
    s1[1] = beta2 * s1[1] + (1 - beta2) * g1 ** 2
    s2[1] = beta2 * s2[1] + (1 - beta2) * g2 ** 2
    m1 = s1[0]/(1-beta1 ** (t+1))
    m2 = s2[0]/(1-beta1 ** (t+1))
    v1 = s1[1]/(1-beta2 ** (t+1))
    v2 = s2[1]/(1-beta2 ** (t+1))
    x1 -= eta / (math.sqrt(v1) + eps) * m1
    x2 -= eta / (math.sqrt(v2) + eps) * m2
    return x1, x2, s1, s2

eta = 0.9

x1, x2 = -5, -2
s1, s2 = [0,0,0], [0,0,0]

res = [(x1, x2)]
print(f'res before : {res}')

for i in range(20):
    x1, x2, s1, s2 = adam_2d(x1, x2, s1, s2, i)
    res.append((x1, x2))

print(f'epoch {i+1}, x1 {x1:.6f}, x2 {x2:.6f}')
print(f'res after : {res}')  
x1, x2 = zip(*res)
print(f'x1 : {x1}')  
print(f'x2 : {x2}')  

plt.plot(x1, x2, '-o', color='red')
x1 = np.arange(min(x1) - 1, max(x1) + 1, 0.1)
x2 = np.arange(min(x2) - 1, max(x2) + 1, 0.1)
x1, x2 = np.meshgrid(x1, x2)
plt.contour(x1, x2, f_2d(x1, x2), colors='blue')
plt.title('Adam') 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 
