# importing necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# We set seeds initially. By doing it so, we can reproduce same results.
tf.random.set_seed(123)

# 100 equidistant points in the domain are created:
x = tf.linspace(0.0, 1.0, 100)

# boundary conditions T(0)=T(1)=0 and \kappa are introduced:
bcs_x = [0.0, 1.0]
print("bcs_x : ", bcs_x)
bcs_T = [0.0, 0.0]
bcs_x_tensor = tf.convert_to_tensor(bcs_x)
print("bcs_x_tensor : ", bcs_x_tensor)
bcs_T_tensor = tf.convert_to_tensor(bcs_T)
kappa = 0.5

# Number of iterations:
N = 1000

# ADAM optimizer with learning rate of 0.01:
optim = tf.keras.optimizers.Adam(learning_rate=0.01)

#The exact solution of the problem:
solution = lambda x: -5 * x**3 + 2 * x**2 + 3 * x

# Function for creating the model:
def buildModel(num_hidden_layers, num_neurons_per_layer):
    tf.keras.backend.set_floatx("float32")
    # Initialize a feedforward neural network:
    model = tf.keras.Sequential()
    # Input is one dimensional ( one spatial dimension):
    model.add(tf.keras.Input(1))
    # Append hidden layers:
    for _ in range(num_hidden_layers):
        model.add(
        tf.keras.layers.Dense(
        num_neurons_per_layer,
        activation=tf.keras.activations.get("tanh"),
        kernel_initializer="glorot_normal",
        )
        )
    # Output is one-dimensional:
    model.add(tf.keras.layers.Dense(1))

    return model

# determine the model size (3 hidden layers with 32 neurons each):
model = buildModel(2, 20)

print(model.summary())

tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, 
                          show_layer_names=True, show_dtype=True, 
                          show_layer_activations=True)

# Boundary loss function:
# @tf.function
def boundary_loss(bcs_x_tensor, bcs_T_tensor):
    predicted_bcs = model(bcs_x_tensor)
    mse_bcs = tf.reduce_mean(tf.square(predicted_bcs - bcs_T_tensor))
    return mse_bcs

# the first derivative of the prediction
def get_first_deriv(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        T = model(x)
    T_x = tape.gradient(T, x)
    return T_x

# the second derivative of the prediction
def second_deriv(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        T_x = get_first_deriv(x)
    T_xx = tape.gradient(T_x, x)
    return T_xx

# Source term divided by \kappa:
source_func = lambda x: (15 * x - 2) / kappa
# def source_func(x): return (15 * x - 2) / kappa

# Function for physics loss:
def physics_loss(x):
    predicted_Txx = second_deriv(x)
    mse_phys = tf.reduce_mean(tf.square(predicted_Txx + source_func(x)))
    return mse_phys

# Overall loss function:
def loss_func(x, bcs_x_tensor, bcs_T_tensor):
    bcs_loss = boundary_loss(bcs_x_tensor, bcs_T_tensor)
    phys_loss = physics_loss(x)
    loss = bcs_loss + phys_loss
    return loss

# taking gradients of the loss function:
def get_grad():
    with tf.GradientTape() as tape:
    # This tape is for derivatives with
    # respect to trainable variables
        tape.watch(model.trainable_variables)
        Loss = loss_func(x, bcs_x_tensor, bcs_T_tensor)
    g = tape.gradient(Loss, model.trainable_variables)
    return Loss, g

# optimizing and updating the weights of the model by using gradients
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = get_grad()
    # Perform gradient descent step
    # Update the weights of the model.
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    return loss

start = time.time()
# Training loop
for i in range(N + 1):
    loss = train_step()
    # printing loss amount in each 100 epoch
    if i % 100 == 0:
        print("Epoch {:03d}: loss = {:10.8e}".format(i, loss))

end = time.time()
computation_time = {}
computation_time["pinn"] = end - start
print(f"\ncomputation time: {end-start:.3f}\n")


plt.plot(x, solution(x)[:, None], label = "Exact Solution", color = "b", linestyle = "-" ) #color='darkorange'
plt.plot(x, model(x), label = "Predicted Solution", color = "r", linestyle = "--" ) #color='navy'
plt.xlabel("x ", fontsize = 12)
plt.ylabel("T(x)", fontsize = 12)
plt.legend(fontsize = 10, loc='best')
# plt.title("1D Heat Transfer", fontsize = 11)
# plt.xlim(xmin = 0, xmax = 1.10) #or plt.xlim([0.0, 1.1])
# plt.ylim(ymin = 0)
# plt.grid()
plt.show()
