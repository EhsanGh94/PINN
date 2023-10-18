"""
PINN Method for the 1D Steady State Heat Equation (Parametric)
# EhsanGh94 & MRS1380
"""

# PyTorch uses a dynamic computational graph that allows for more flexible model building and experimentation.
# importing necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn  # neural networks
from torch.autograd import grad
import time
# from torch.utils.data import TensorDataset, DataLoader
# ! pip install pyDOE
from pyDOE import lhs  #Latin Hypercube Sampling

# Device configuration:
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# We set seeds initially. By doing it so, we can reproduce same results.
torch.manual_seed(1234)
np.random.seed(1234)

x_min = 0.0
x_max = 1.0
kappa_min = 0.5
kappa_max = 1.5
ub = np.array([x_max, kappa_max])
lb = np.array([x_min, kappa_min])
N_b = 100
N_c = 100

def getData():

    x_ik = [x_min, kappa_min] + [0.0, kappa_max - kappa_min] * lhs(2, N_b)
    x_i = np.zeros((N_b, 1))
    x_k = x_ik[:,1:2]
    x_ik = np.concatenate([x_i, x_k], axis=1)
    T_i = np.zeros((N_b, 1))
    x_ok = [x_max, kappa_min] + [0.0, kappa_max - kappa_min] * lhs(2, N_b)
    x_o = np.ones((N_b, 1))
    x_k = x_ok[:,1:2]
    x_ok = np.concatenate([x_o, x_k], axis=1)
    T_o = np.zeros((N_b, 1))

    xk_bnd = np.concatenate([x_ik, x_ok], axis=0)
    T_bnd = np.concatenate([T_i, T_o], axis=0)

    xk_col = lb + (ub - lb) * lhs(2, N_c)
    print(xk_col.shape)

    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # plt.scatter(x, x_col, marker='o', alpha=0.4 ,color='blue')
    # plt.scatter(x, x_i, marker='o', alpha=0.5 , color='green')
    # plt.scatter(x, x_o, marker='o', alpha=0.5, color='orange')
    # plt.show()

    xk_bnd = torch.tensor(xk_bnd, dtype=torch.float32).to(device)
    T_bnd = torch.tensor(T_bnd, dtype=torch.float32).to(device)
    xk_col = torch.tensor(xk_col, dtype=torch.float32).to(device)

    return xk_col, xk_bnd, T_bnd

xk_col, xk_bnd, T_bnd = getData()

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
        self.net = DNN(dim_in=2, dim_out=1, n_layer=2, n_node=20, ub=ub, lb=lb).to(
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

    def predict(self, xk):
        out = self.net(xk)
        KAPPA = xk[:,1:2]
        T = out
        return T, KAPPA

    def bc_loss(self, xk):
        T = self.net(xk)
        mse_bc = torch.mean(torch.square(T - T_bnd))
        return mse_bc

    def pde_loss(self, xk):
        xk = xk.clone()
        xk.requires_grad = True
        T, KAPPA = self.predict(xk)
        T_out = grad(T.sum(), xk, create_graph=True)[0]
        T_x = T_out[:, 0:1]
        T_out2 = grad(T_x.sum(), xk, create_graph=True)[0]
        T_xx = T_out2[:, 0:1]
        f0 = T_xx + ((1/KAPPA)*(15*xk[:,0:1]-2))
        mse_f0 = torch.mean(torch.square(f0))
        mse_pde = mse_f0
        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()
        mse_bc = self.bc_loss(xk_bnd)
        mse_pde = self.pde_loss(xk_col)
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

kappa_g = 0.5
kappa = np.arange(kappa_g, kappa_g+0.00001, 0.001)
x = np.arange(x_min, x_max+0.01, 0.01)
solution = lambda x: -5 * x**3 + 2 * x**2 + 3 * x
X, KAPPA= np.meshgrid(x, kappa)
x = X.reshape(-1, 1)
kappa = KAPPA.reshape(-1, 1)
xk = np.concatenate([x, kappa], axis=1)
xk = torch.tensor(xk, dtype=torch.float32).to(device)
with torch.no_grad():
    T, KAPPA = pinn.predict(xk)
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
