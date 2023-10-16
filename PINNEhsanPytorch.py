"""
PINN Method for the 1D Steady State Heat Equation
#EhsanGh94
"""

# PyTorch uses a dynamic computational graph that allows for more flexible model building and experimentation.
# importing necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
from torch.autograd import grad
import time
# from torch.utils.data import TensorDataset, DataLoader
# ! pip install pyDOE
from pyDOE import lhs

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# We set seeds initially. By doing it so, we can reproduce same results.
torch.manual_seed(1234)
np.random.seed(1234)

x_min = 0.0
x_max = 1.0
ub = np.array([x_max])
lb = np.array([x_min])
N_b = 100
N_c = 100

def getData():

    x_i = np.zeros((N_b, 1))
    T_i = np.zeros((N_b, 1))
    x_o = np.ones((N_b, 1))
    T_o = np.zeros((N_b, 1))

    x_bnd = np.concatenate([x_i,x_o], axis=0)
    T_bnd = np.concatenate([T_i, T_o], axis=0)

    x_col = lb + (ub - lb) * lhs(1, N_c)
    print(x_col.shape)

    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # plt.scatter(x, x_col, marker='o', alpha=0.4 ,color='blue')
    # plt.scatter(x, x_i, marker='o', alpha=0.5 , color='green')
    # plt.scatter(x, x_o, marker='o', alpha=0.5, color='orange')
    # plt.show()

    x_bnd = torch.tensor(x_bnd, dtype=torch.float32).to(device)
    T_bnd = torch.tensor(T_bnd, dtype=torch.float32).to(device)
    x_col = torch.tensor(x_col, dtype=torch.float32).to(device)

    return x_col, x_bnd, T_bnd

x_col, x_bnd, T_bnd = getData()

def plotLoss(losses_dict, path, info=["B.C.", "P.D.E."]):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(2), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class layer(nn.Module):

    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

class DNN(nn.Module):

    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        out = x
        for layer in self.net:
            out = layer(out)
        return out

class PINN:

    def __init__(self) -> None:
        self.net = DNN(dim_in=1, dim_out=1, n_layer=2, n_node=20, ub=ub, lb=lb).to(
            device
        )

        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=500,
            max_eval=500,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self.adam = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.losses = {"bc": [], "pde": []}
        self.iter = 0

    def predict(self, x):
        out = self.net(x)
        T = out
        return T

    def bc_loss(self, x):
        T = self.predict(x)
        mse_bc = torch.mean(torch.square(T - T_bnd))
        return mse_bc

    def pde_loss(self, x):
        x = x.clone()
        x.requires_grad = True
        T = self.predict(x)
        T_out = grad(T.sum(), x, create_graph=True)[0]
        T_x = T_out[:, 0:1]
        T_out2 = grad(T_x.sum(), x, create_graph=True)[0]
        T_xx = T_out2[:, 0:1]
        f0 = T_xx + (30*x-4)
        mse_f0 = torch.mean(torch.square(f0))
        mse_pde = mse_f0
        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()
        mse_bc = self.bc_loss(x_bnd)
        mse_pde = self.pde_loss(x_col)
        loss = mse_bc + mse_pde
        loss.backward()
        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.iter += 1
        print(
            f"\r It: {self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e}  pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 100 == 0:
            print("")
        return loss

if __name__ == "__main__":
    pinn = PINN()
    start_time = time.time()
    for i in range(500):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f'-- {(time.time() - start_time)/60} mins --')
    torch.save(pinn.net.state_dict(), "/content/Param.pt")
    plotLoss(pinn.losses, "/content/LossCurve.png", ["BC", "PDE"])

pinn = PINN()
pinn.net.load_state_dict(torch.load("/content/Param.pt"))

x = np.arange(x_min, x_max+0.01, 0.01)
solution = lambda x: -5 * x**3 + 2 * x**2 + 3 * x
x = x.reshape(-1, 1)
x = torch.tensor(x, dtype=torch.float32).to(device)
with torch.no_grad():
    T = pinn.predict(x)
    T = T.cpu().numpy()
    T = T.reshape(x.shape)
x = np.arange(x_min, x_max+0.01, 0.01)
plt.plot(x, solution(x), label = "Exact Solution", color = "b", linestyle = "-" ) #color='darkorange'
plt.plot(x, T, label = "Predicted Solution", color = "r", linestyle = "--" ) #color='navy'
plt.xlabel("x ", fontsize = 12)
plt.ylabel("T(x)", fontsize = 12)
plt.legend(fontsize = 10, loc='best')
# plt.title("1D Heat Transfer", fontsize = 11)
# plt.xlim(xmin = 0, xmax = 1.10) #or plt.xlim([0.0, 1.1])
# plt.ylim(ymin = 0)
# plt.grid()
plt.show()
