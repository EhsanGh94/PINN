print("Hello", "World!")

i = 4                                        # Integer
f = 3.14                                     # Float
s1 = 'string variable with single qoutes'    # String
s2 = "string variable with double qoutes"    # String
b = True                                     # Boolean
c = 2 + 6j                                   # Complex
print('The type of variable s1 is', type(s1))
print(isinstance(i, int))
#print(type(i) == int)

l = [1,2,3,4]                      # List
#l = list((1,2,3,4))
list1 = ["abc", 22, True, 50, "Ehsan"]
print('The number of items in the list is', len(list1))
t = (1,2,3,4)                      # Tuple
#t = tuple((1,2,3,4))
s = {1,2,3,4}                      # Set
#s = set((1,2,3,4))
d = {'Ali': 24, 'Mohammad': 29}    # Dictionary
#d = dict('Ali'=24, 'Mohammad'=29)
print('The type of variable t is', type(t))

# Addition
print('5 + 3 =', 5 + 3)
# Subtraction
print('5 - 3 =', 5 - 3)
# Multiplication
print('5 * 3 =', 5 * 3)
# Exponentiation
print('2 ^ 5 =', 2 ** 5)
#print('2 ^ 5 =', pow(2,5))
# Division
print('13 / 3 =', 13 / 3)
# Floor division (the floor division // rounds the result down to the nearest whole number)
print('15 // 2 =', 15 // 2)
# Modulus
print('5 % 2 =', 5 % 2)

a = 3
b = 530
print("A") if a > b else print("B")
if a > b:
    print("A") 
# elif
else:
    print("B")

i = 1
while i < 6:
    print(i)
    if i == 3:
        break
    i += 1

fruits = ["apple", "banana", "cherry"]
for x in fruits:
    if x == "banana":
        continue
    print(x)
for x in range(2, 30, 3):
    print(x)

def my_function(x):
    return 5 * x
print(my_function(3))

##############################################################################################################################
##############################################################################################################################

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NumPy is short for "Numerical Python".
anumpy = np.array([1, 2, 3, 4])
print('anumpy', anumpy)
print('type(anumpy)', type(anumpy))
print('np.arange', np.arange(60,75,5))
print('np.linspace', np.linspace(11,23,3))
arr2D = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('arr2D[0:2, 2]', arr2D[0:2, 2])
arr3D = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print('arr3D.shape', arr3D.shape)
print('arr3D.reshape(-1)', arr3D.reshape(-1))
for x in arr3D:
    print('x', x)
for x in arr3D:
    for y in x:
        for z in y:
            print('z', z)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print('np.where', np.where(arr%2 == 1))

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print('np.concatenate', np.concatenate((arr1, arr2)))

##############################################################################################################################

apandas = pd.Series([1, 2, 3, 4])
print('apandas', apandas)
apndas2 = pd.DataFrame({'a': [11, 21, 31], 'b': [12, 22, 312]}) # index=  ## A DataFrame is a table.
print('apndas2', apndas2)
apndas2.to_csv("apndas2.csv") ## Comma-Separated Values
# apndas2.to_excel("apndas2.xlsx")
apndas2 = pd.read_csv("apndas2.csv")
# apndas2.head()
# apndas2.iloc[0]
print(apndas2.isnull().sum())

##############################################################################################################################

print(torch.__version__)
print(torch.cuda.is_available())
# the CUDA tool is a development environment for creating high performance GPU accelerated aplications
# that for this you need an NVIDIA GPU on your machine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device : ', device)

# Set random seed
torch.manual_seed(1)

a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)  # torch.int32
# torch.tensor([1,2,3], device='cpu')
# x = torch.tensor(3, requires_grad=True, dtype=torch.float32)
# x.detach() ## This allows us to then convert the tensor to a numpy array.
# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('a.max()', a.max())
print('a.mean()', a.mean())
print('a[0]', a[0])
print('a[0].item()', a[0].item())  # y.data
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('a[0][1]', a[0][1])  # a[1:3, 1] != a[1:3][1]
print('a[0, 1]', a[0, 1])
print('a[0, :]', a[0, :])
print('a[:-1, 2]', a[:-1, 2])
print('a', a)
print('a.dtype', a.dtype)
print('a.type()', a.type())
print('a.size()', a.size())
print('a.shape', a.shape)
print('a.ndimension()', a.ndimension())
# b = a.reshape(4,-1)
# same storage => use view
# a.view(4, -1)
# Whenever we want to make a copy of a tensor and ensure that any operations are done with the cloned tensor to ensure that the gradients are propagated to the original tensor, we must use clone().
print('b', b)
a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)  # torch.int32
print('id(a)', id(a))
# print('id(a.reshape(4,-1))', id(a.reshape(4,-1)))
print('id(a.view(4, -1))', id(a.view(4, -1)))
print('id(a.clone())', id(a.clone()))

x1 = torch.rand(1, 3)
x2 = torch.randn(4)
y1 = torch.arange(1, 4, dtype=torch.float32)  #torch.arange(1,3, step=0.5)
y2 = torch.linspace(-2, 2, steps=5)
yy = torch.arange(1,10).view(3,-1)
a = [i for i in range(6)]
a1 = torch.tensor(a).view(2,-1)
b1 = torch.concat([a1,a1,a1], dim = 0)
print('a1', a1)
print('b1', b1)
x = torch.linspace(-10, 10, 1000, requires_grad=True)
Y = torch.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.xlabel('x')
plt.legend()
plt.show()

'''
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
Q.sum().backward() ## # Q.backward() !error
# sum applies and .grad holds the sum of derivatives with different values
'''

# Basic Autograd Example:
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)
y = w * x + b    # y = 2 * x + 3

# Compute gradients:
y.backward()

print('dy/dx:', x.grad)    # x.grad = 2 
print('dy/dw:', w.grad)    # w.grad = 1 
print('dy/db:', b.grad)    # b.grad = 1 

# linear Regression:
from torch.nn import Linear

lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))
print("lr.state_dict(): ", lr.state_dict())
# change model parameters
lr.weight.data = torch.tensor([[0.2]])
print("lr.state_dict(): ", lr.state_dict())

from torch import nn

# Customize Linear Regression Class
class LR(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out

lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print()
print("Linear model: ", lr.linear)
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)
