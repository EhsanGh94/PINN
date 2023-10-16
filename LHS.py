# Latin hypercube sampling (LHS) is a statistical method for generating a near random samples with equal intervals.
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(181)

def latin_hypercube_2d_uniform(n):
    lower_limits=np.arange(0,n)/n
    upper_limits=np.arange(1,n+1)/n

    points=np.random.uniform(low=lower_limits,high=upper_limits,size=[2,n]).T
    np.random.shuffle(points[:,1])
    return points

n=20
p=latin_hypercube_2d_uniform(n)
plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(p[:,0],p[:,1],c='r')

for i in np.arange(0,1,1/n):
    plt.axvline(i)
    plt.axhline(i)
# plt.grid()
plt.show()


###################

lll = np.arange(0,n)/n
print('lll : ', lll )

uuu = np.arange(1,n+1)/n
print('uuu : ', uuu )

ppp = np.random.uniform(low=lll,high=uuu,size=[2,n]).T
print('ppp : ', ppp)

pezaf = np.random.shuffle(ppp[:,1])
print('pezaf : ', ppp)


'''
# Get the uniform random samples
arr = np.random.uniform(low = 3, high = 5, size = 5)
print(arr)
# Output :
# [3.87875762 4.20867489 4.13948288 3.28137881 4.4622086 ]


# Get the random sample of multi-Dimensional array
arr = np.random.uniform(low = 3, high = 5, size = (2, 3))
print(arr)
# Output :
# [[3.18967671 3.93263376 4.31706326]
# [3.85327017 4.22567335 4.91062538]]
'''
arr = np.random.uniform(low = 3, high = 5, size = (2, 3))
print(arr)

###############################################################################################

# ! pip install pyDOE
# ! pip install scikit-optimize

from pyDOE import *

np.random.seed(181)

lhd = lhs(2, samples=20)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(lhd[:,0],lhd[:,1],c='r')

for i in np.arange(0,1,1/n):
    plt.axvline(i)
    plt.axhline(i)
# plt.grid()
plt.show()


###############################################################################################

from skopt.space import Space
from scipy.spatial.distance import pdist
from skopt.sampler import Lhs


def plot_searchspace(x, title):
    fig, ax = plt.subplots()
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', label='samples')
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', markersize=80, alpha=0.5)
    # ax.legend(loc="best", numpoints=1)
    ax.set_xlabel("X1")
    ax.set_xlim([-5, 10])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 15])
    plt.title(title)
    plt.show()

n_samples = 10
space = Space([(-5., 10.), (0., 15.)])
# space.set_transformer("normalize")

pdist_data = []
x_label = []

# x = space.rvs(n_samples)
# plot_searchspace(x, "Random samples")
# pdist_data.append(pdist(x).flatten())
# x_label.append("random")

lhs = Lhs(lhs_type="classic", criterion=None)
x = lhs.generate(space.dimensions, n_samples)
plot_searchspace(x, 'classic LHS')
pdist_data.append(pdist(x).flatten())
x_label.append("lhs")


#######################
#######################
#######################


n = 20
space = Space([np.arange(0,n)/n, np.arange(1,n+1)/n])

lhs = Lhs(lhs_type="classic", criterion=None)
x = lhs.generate(space.dimensions, n)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(np.array(x)[:,0],np.array(x)[:,1],c='r')

# for i in np.arange(0,1,1/n):
#     plt.axvline(i)
#     plt.axhline(i)
plt.grid()
plt.title("classic")
plt.show()




#######################

x = space.rvs(n)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(np.array(x)[:,0],np.array(x)[:,1],c='r')

# for i in np.arange(0,20,20/n):
#     plt.axvline(i)
#     plt.axhline(i)
plt.grid()
plt.title("Random samples")
plt.show()


#######################

from skopt.sampler import Sobol

sobol = Sobol()
x = sobol.generate(space.dimensions, n)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(np.array(x)[:,0],np.array(x)[:,1],c='r')

# for i in np.arange(0,20,20/n):
#     plt.axvline(i)
#     plt.axhline(i)
plt.grid()
plt.title("sobol")
plt.show()


#######################

lhs = Lhs(criterion="maximin", iterations=10000)
x = lhs.generate(space.dimensions, n)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(np.array(x)[:,0],np.array(x)[:,1],c='r')

# for i in np.arange(0,20,20/n):
#     plt.axvline(i)
#     plt.axhline(i)
plt.grid()
plt.title("maximin")
plt.show()


#######################

lhs = Lhs(criterion="ratio", iterations=10000)
x = lhs.generate(space.dimensions, n)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(np.array(x)[:,0],np.array(x)[:,1],c='r')

# for i in np.arange(0,20,20/n):
#     plt.axvline(i)
#     plt.axhline(i)
plt.grid()
plt.title("ratio")
plt.show()



#######################

from skopt.sampler import Halton

halton = Halton()
x = halton.generate(space.dimensions, n)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(np.array(x)[:,0],np.array(x)[:,1],c='r')

# for i in np.arange(0,20,20/n):
#     plt.axvline(i)
#     plt.axhline(i)
plt.grid()
plt.title("halton")
plt.show()


#######################

from skopt.sampler import Hammersly

hammersly = Hammersly()
x = hammersly.generate(space.dimensions, n)

plt.figure(figsize=[5,5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(np.array(x)[:,0],np.array(x)[:,1],c='r')

# for i in np.arange(0,20,20/n):
#     plt.axvline(i)
#     plt.axhline(i)
plt.grid()
plt.title("hammersly")
plt.show()
