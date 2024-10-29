import numpy as np
import sys
#from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import matplotlib.patheffects as patheffects


PI = np.pi
SIN = np.sin


class LinearSinModel(nn.Module):
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(6)
        self.linear = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
        #     # randomize the initial omega
            self.linear.weight.copy_(torch.randn(2, 1))
        #     self.linear.weight.copy_(torch.tensor([[1.], [1.1]]))

    def forward(self, x):
        a = self.linear(x)
        # Apply sine to each prediction element-wise, then sum the outputs
        sin_a = torch.sin(a)
        # Ensure dimensions match y_true
        y_pred = torch.sum(sin_a, dim=1, keepdim=True)
        return y_pred


class BinarizeParams(torch.autograd.Function):
    """
    Binarize the weights, biases, and activations of a neural network
    """
    @staticmethod
    def forward(ctx, input):
        condition_greater = input > 1e-6
        condition_less = input < -1e-6
        result_tensor = torch.where(
            condition_greater, torch.tensor(1.),
            torch.where(condition_less, torch.tensor(-1.), torch.tensor(0.)))
        return result_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # saturated Straight-Through Estimator (STE)
        grad_input = grad_output.clone()
        return grad_input


class BinarizedSinModel(nn.Module):
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(6)
        # self.weight = nn.Parameter(
        #         torch.randint(0, 2, (2, 1)) * 2.0 - 1)
        self.weight = nn.Parameter(
                torch.rand(2, 1) * 4 - 2)

    def forward(self, x):
        weight_b = BinarizeParams.apply(self.weight)
        linear_output = F.linear(x, weight_b)
        sin_output = torch.sin(linear_output)
        sin_output_b = BinarizeParams.apply(sin_output)
        y_pred = torch.sum(sin_output_b, dim=1, keepdim=True)
        y_pred_b = BinarizeParams.apply(y_pred)
        return y_pred_b


# Custom loss function
def custom_loss(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2) 
#+ torch.randn(1) * 0.02


def compute_hessian_and_eigenvalues(model, data, target):
    """
    Compute the Hessian matrix and its eigenvalues for the weights of a neural
    network model.

    :param model: The neural network model.
    :param data: Input data (X).
    :param target: Target data (Y).
    :return: Hessian matrix and its eigenvalues.
    """
    # Forward pass
    output = model(data)
    # Compute loss
    loss = torch.mean(
            (target - torch.sum(torch.sin(output), dim=1, keepdim=True)) ** 2)

    # First-order gradients (w.r.t weights)
    first_order_grads = torch.autograd.grad(
            loss, model.parameters(), create_graph=True)

    # Flatten the first-order gradients
    grads_flatten = torch.cat(
            [g.contiguous().view(-1) for g in first_order_grads])

    # Hessian computation
    hessian = []
    for grad in grads_flatten:
        # Compute second-order gradients
        # (w.r.t each element in the first-order gradients)
        second_order_grads = torch.autograd.grad(
                grad, model.parameters(), retain_graph=True)

        # Flatten and collect the second-order gradients
        hessian_row = torch.cat(
                [g.contiguous().view(-1) for g in second_order_grads])
        hessian.append(hessian_row)

    # Stack to form the Hessian matrix
    hessian_matrix = torch.stack(hessian)

    # Compute eigenvalues
    eigenvalues, _ = torch.linalg.eig(hessian_matrix)

    return hessian_matrix, eigenvalues

# Note: To use this function, you'll need to provide your neural network model,
# the input data (X), and the target data (Y).


def check_local_minimum(eigenvalues):
    # Check if all eigenvalues have a positive real part
    if all(eig.real > 0 for eig in eigenvalues):
        print("This is a local minimum.")
    else:
        print("This is not a local minimum.")


def train_model(
        seed, inputs, targets, num_epochs=100, lr=0.05, print_every=10):
    # Initialize the model and optimizer with the given seed
    model = LinearSinModel(seed=seed)
    if type(model) is LinearSinModel:
        weights = model.linear.weight
    elif type(model) in [BinarizedSinModel,
]:
        weights = model.weight
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Dictionary to store the loss history
    loss_history = []

    weights1 = round(weights.data.numpy()[0][0], 5)
    weights2 = round(weights.data.numpy()[1][0], 5)
    print(f"Epoch {0:4}/{num_epochs}, Loss: {'----':4}, "
          f"Weights: {weights1:6.3f}, {weights2:6.3f}")
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = custom_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Record the loss
        loss_history.append(loss.item())

        # Print every 'print_every' epochs
        if (epoch + 1) % print_every == 0:
            weight1 = round(weights.data.numpy()[0][0], 5)
            weight2 = round(weights.data.numpy()[1][0], 5)
            gradient1 = round(weights.grad.data.numpy()[0][0], 5)
            gradient2 = round(weights.grad.data.numpy()[1][0], 5)
            print(f"Epoch {epoch+1:4}/{num_epochs}, Loss: {loss.item():.2f}, "
                  f"Weights: {weight1:6.3f}, {weight2:6.3f},  "
                  f"Gradients: {gradient1:6.3f}, {gradient2:6.3f}")

        # if epoch + 1 == num_epochs:
        #     (hessian_matrix_central,
        #      eigenvalues_central) = compute_hessian_and_eigenvalues(
        #              model, inputs, targets)
        #     print(eigenvalues_central)
        #     check_local_minimum(eigenvalues_central)
    return loss_history, weights.data.numpy()


# Data for training
inputs = torch.tensor(
        [
            [-3*PI/2], [-PI/2],
            [PI/2], [3*PI/2],
         ], dtype=torch.float32)

targets = torch.tensor(
        [
            [2.], [-2.],
            [2.], [-2.],
        ], dtype=torch.float32)
# inputs = torch.tensor(
#         [
#             [-1.],
#             [1.],
#          ], dtype=torch.float32)
# targets = torch.tensor(
#         [
#             [-1.],
#             [1.],
#         ], dtype=torch.float32)

loss_histories = {}
weights = {}

# Train the model for different seeds
for i in [x for x in range(30, 51) if x != 32]:  # Adjust the range for more seeds
    print(f"Training with seed {i}")
    loss_history, weight = train_model(
            seed=i, inputs=inputs, targets=targets,
            num_epochs=30, lr=0.01, print_every=10)
    loss_histories[i] = loss_history
    weights[i] = weight

# Convert each value in the dictionary to a list
final_data_serializable = {k: v.tolist() for k, v in weights.items()}

# Save the final dictionary to a text file
with open('final_weights.txt', 'w') as file:
    json.dump(final_data_serializable, file)

print("Dictionary saved successfully.")



# Example of how to access and print the loss history for a specific seed
# print(len(loss_histories[25]))  # Length of loss history for seed 5
# print(loss_histories[25])  # Loss history for seed 5
plt.rcParams.update({
    'text.usetex': True,             # Use LaTeX for all text rendering
    'font.family': 'serif',          # Set the font family to serif
    'font.serif': ['Computer Modern'], # Explicitly use Computer Modern
})
plt.figure(figsize=(20, 10))
for seed, history in loss_histories.items():
    plt.plot(history, label=f'Seed {seed}', linewidth=4)

plt.xlabel('Epoch', fontsize=46, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
plt.ylabel('SinNN Loss', fontsize=46, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
plt.xticks(fontsize=28, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
plt.yticks(fontsize=28, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
#plt.legend(bbox_to_anchor=(1.002, 1), loc='upper left', fontsize=18, borderaxespad=0.)
# Adjust the layout to prevent cutting off the legend
plt.subplots_adjust(right=0.85)
# plt.title('Loss History by Seed (Binarized Weights & Activations)')

# Place the legend on the left side outside of the plot
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

# Save the plot as a PNG file (you can change the format if needed)
plt.savefig("SinNN_Loss_Epoch.png", dpi=400, bbox_inches='tight')

plt.show()


# hessian_matrix_central, eigenvalues_central = compute_hessian_and_eigenvalues(model, inputs, targets)

# print(eigenvalues_central)
# check_local_minimum(eigenvalues_central)
