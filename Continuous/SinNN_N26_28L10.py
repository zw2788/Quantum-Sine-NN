import numpy as np
import sys
#from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import matplotlib.patheffects as patheffects
import math
import os
import csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global plot settings
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
})


PI = np.pi
SIN = np.sin


class LinearSinModel(nn.Module):
    def __init__(self, seed=None, num_neuron=None, num_hidden_layers=1, mode='constant', w_0 = 10):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(6)

        self.mode = mode
        self.w_0 = w_0
        self.linear = nn.Linear(1, num_neuron, bias=False)

        self.hidden_layers = nn.ModuleList()
        current_neurons = num_neuron

        for i in range(max(num_hidden_layers - 1, 0)):
            if mode == 'constant':
                next_neurons = current_neurons
            elif mode == 'decreasing':
                if current_neurons > 5:
                    next_neurons = current_neurons - 1
                else:
                    next_neurons = 5  # stay at 2 once reached
            else:
                raise ValueError("Mode must be 'constant' or 'decreasing'")

            self.hidden_layers.append(nn.Linear(current_neurons, next_neurons, bias=False))
            current_neurons = next_neurons


        with torch.no_grad():
            # origin weight distribution setting
            # self.linear.weight.copy_(torch.rand(num_neuron, 1) * 4)
            # new uniform
            #self.linear.weight.copy_(torch.empty(num_neuron, 1).uniform_(0, 4))
            
            self.linear.weight.copy_(torch.empty(num_neuron, 1).uniform_(-1.0, 1.0))

            # origin weight distribution setting
            # for hl in self.hidden_layers:
            #     hl.weight.copy_(torch.rand(hl.out_features, hl.in_features) * 100)
            # new uniform
            # for hl in self.hidden_layers:
            #     hl.weight.copy_(torch.empty(hl.out_features, hl.in_features).uniform_(-50, 50))
            for hl in self.hidden_layers:
                fan_in = hl.in_features
                bound = math.sqrt(6 / fan_in)
                hl.weight.copy_(
                 torch.empty(hl.out_features, hl.in_features).uniform_(-bound, bound))

        print(f"Created model with {len(self.hidden_layers)} extra hidden layers.")

        

    def forward(self, x):
        out = torch.sin(self.w_0 * self.linear(x))
        # Apply sine to first layer
        # extra hidden layer
        for i, hl in enumerate(self.hidden_layers):
            out = hl(out)  # wx
            out = torch.sin(out)
            #out = torch.sin(out)  # Apply sin(wx)

        y_pred = torch.sum(out, dim=1, keepdim=True)


        return y_pred
    

# ========== Binary Function ==========
class BinarizeWeightParams(torch.autograd.Function):
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
    
# ========== Binary Function ==========
class BinarizeSinParams(torch.autograd.Function):
    """
    Binarize the weights, biases, and activations of a neural network
    """
    @staticmethod
    def forward(ctx, input):
        condition_greater = input > 1e-2
        condition_less = input < -1e-2
        result_tensor = torch.where(
            condition_greater, torch.tensor(1.),
            torch.where(condition_less, torch.tensor(-1.), torch.tensor(0.)))
        return result_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # saturated Straight-Through Estimator (STE)
        grad_input = grad_output.clone()
        return grad_input

# ========== Binary Sinusoidal Model ==========
class BinarizedSinModel(nn.Module):
    def __init__(self, seed=None, num_neuron=4, num_hidden_layers=1, mode='constant', w_0=10):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(6)

        self.mode = mode
        self.w_0 = w_0
        self.linear = nn.Linear(1, num_neuron, bias=False)
        self.hidden_layers = nn.ModuleList()
        current_neurons = num_neuron

        for i in range(max(num_hidden_layers - 1, 0)):
            if mode == 'constant':
                next_neurons = current_neurons
            elif mode == 'decreasing':
                next_neurons = max(2, current_neurons - 1)
            else:
                raise ValueError("Mode must be 'constant' or 'decreasing'")

            self.hidden_layers.append(nn.Linear(current_neurons, next_neurons, bias=False))
            current_neurons = next_neurons

        # Initialize weights (not really used after binarization)
        with torch.no_grad():
            self.linear.weight.copy_(torch.empty(num_neuron, 1).uniform_(-1, 1))
            for hl in self.hidden_layers:
                fan_in = hl.in_features
                bound = math.sqrt(6 / fan_in)
                hl.weight.copy_(torch.empty(hl.out_features, hl.in_features).uniform_(-bound, bound))

        print(f"Created BinarySinNN with {len(self.hidden_layers)} hidden layers.")

    def forward(self, x):
        # First layer: input -> binarized linear -> sin -> binarized
        weight_b = BinarizeWeightParams.apply(self.linear.weight)
        out = F.linear(x, weight_b)
        out = torch.sin(self.w_0 * out+ 0.1)
        out = BinarizeSinParams.apply(out)


        # Hidden layers
        for hl in self.hidden_layers:
            weight_b = BinarizeWeightParams.apply(hl.weight)
            fan_in = hl.in_features
            m = math.floor(math.log2(fan_in))
            scale = math.pi / (2 ** m)
            out = F.linear(out, scale*weight_b)
            out = torch.sin(out + 0.1)
            out = BinarizeSinParams.apply(out)

        # Final output aggregation
        y_pred = torch.sum(out, dim=1, keepdim=True)
        # tenary DSinNN
        # y_pred_b = BinarizeSinParams.apply(y_pred)
        return y_pred
    
def ternarize_tensor(y, threshold=1e-6):
    condition_greater = y > threshold
    condition_less = y < -threshold
    return torch.where(condition_greater, torch.tensor(1.),
                       torch.where(condition_less, torch.tensor(-1.), torch.tensor(0.)))
    


# class BinarizedSinModel(nn.Module):
#     def __init__(self, seed=None):
#         super().__init__()
#         if seed is not None:
#             torch.manual_seed(seed)
#         else:
#             torch.manual_seed(6)
#         # self.weight = nn.Parameter(
#         #         torch.randint(0, 2, (2, 1)) * 2.0 - 1)
#         self.weight = nn.Parameter(
#                 torch.rand(2, 1) * 4 - 2)

#     def forward(self, x):
#         weight_b = BinarizeWeightParams.apply(self.weight)
#         linear_output = F.linear(x, weight_b)
#         sin_output = torch.sin(linear_output)
#         sin_output_b = BinarizeSinParams.apply(sin_output)
#         y_pred = torch.sum(sin_output_b, dim=1, keepdim=True)
#         y_pred_b = BinarizeSinParams.apply(y_pred)
#         return y_pred_b


# Custom loss function
def custom_loss(y_pred, y_true):
    # MSE
    return torch.mean((y_true - y_pred) ** 2)
    




def normalize_train_test_inputs(train_inputs, test_inputs):
    # to check the full normalization range
    all_inputs = torch.cat([train_inputs, test_inputs], dim=0)
    min_val = all_inputs.min()
    max_val = all_inputs.max()

    # normalize to [-1, 1]
    def normalize(x):
        return 2 * (x - min_val) / (max_val - min_val) - 1

    train_inputs_norm = normalize(train_inputs)
    test_inputs_norm = normalize(test_inputs)

    return train_inputs_norm, test_inputs_norm, min_val.item(), max_val.item()





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
        seed, num_neuron, num_hidden_layers, mode, inputs, targets, num_epochs=100, lr=0.05, print_every=10, w_0=10):
    # Initialize the model and optimizer with the given seed
    model =LinearSinModel(seed=seed, num_neuron=num_neuron, num_hidden_layers=num_hidden_layers, mode= mode, w_0= w_0)
    if type(model) is LinearSinModel:
        weights = model.linear.weight
    # elif type(model) is InverseLinearSinModel:
    #     weights = model.linear.weight
    elif type(model) in [BinarizedSinModel,
]:
        # weights = model.weight
        weights = model.linear.weight

    # SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Adam
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        # Update parameters_original
        # optimizer.step()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    return loss_history, weights.data.numpy(), model


# Test the trained model
def test_model(model, test_inputs, test_targets):
    with torch.no_grad():
        test_outputs = model(test_inputs)
        test_loss = custom_loss(test_outputs, test_targets)
        print(f"Test Loss: {test_loss.item():.6f}")
    return test_loss.item()

def plot_loss_histories(loss_histories, filename, xlabel, ylabel, title=None, plot_type='line'):
    plt.figure(figsize=(20, 10))
    
    if plot_type == 'scatter':
        seeds = list(loss_histories.keys())
        losses = [v[0] for v in loss_histories.values()]
        plt.scatter(seeds, losses, c='b', s=50)
    
    elif plot_type == 'bar':
        seeds = list(loss_histories.keys())
        losses = [v[0] for v in loss_histories.values()]
        plt.bar(seeds, losses, color='b')
    
    else:  # Default to line plot
        for seed, history in loss_histories.items():
            plt.plot(history, label=f'Seed {seed}', linewidth=4)

    plt.xlabel(xlabel, fontsize=46, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
    plt.ylabel(ylabel, fontsize=46, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
    plt.xticks(fontsize=28, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
    plt.yticks(fontsize=28, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
    if title:
        plt.title(title, fontsize=36)

    plt.subplots_adjust(right=0.85)
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    print(f"Plot saved as {filename}")


def generate_random_split(seed=42, start = -7 * PI/2, end= 7 * PI/2, train_ratio=0.8, total_points=100):

    all_inputs = torch.linspace(start, end, steps=total_points).reshape(-1, 1)
    all_targets = 2 * torch.sin(all_inputs)

    num_train = int(train_ratio * total_points)

    # shuffle the indices
    g = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_points, generator=g)

    train_indices = shuffled_indices[:num_train]
    test_indices = shuffled_indices[num_train:]

    train_inputs = all_inputs[train_indices]
    train_targets = all_targets[train_indices]

    test_inputs = all_inputs[test_indices]
    test_targets = all_targets[test_indices]

    return train_inputs, train_targets, test_inputs, test_targets






# Data for training
# inputs = torch.tensor(
#         [
#             [-3*PI/2], [-PI/2],
#             [PI/2], [3*PI/2],
#          ], dtype=torch.float32)


# targets = torch.tensor(
#         [
#             [2.], [-2.],
#             [2.], [-2.],
#         ], dtype=torch.float32)


# multilayer target

#targets= 2 * torch.sin(2 * torch.sin(inputs))

# test_inputs = torch.tensor(
#         [
#             [-7*PI/2], [-5*PI/2],
#             [5*PI/2], [7*PI/2],
#          ], dtype=torch.float32)

# test_targets = torch.tensor(
#         [
#             [2.], [-2.],
#             [2.], [-2.],
#         ], dtype=torch.float32)

# Binary NN
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


# For Linear Model
inputs, targets, test_inputs, test_targets = generate_random_split(seed=42, start = -7 * PI/2, end= 7 * PI/2, train_ratio=0.8, total_points=200)

train_inputs_norm, test_inputs_norm, min_val, max_val = normalize_train_test_inputs(inputs, test_inputs)

# For Discrete Model
# inputs, targets, test_inputs, test_targets = generate_random_split(seed=42, start=-7* PI/2, end=7 * PI/2, train_ratio=0.8, total_points=100)
# targets = ternarize_tensor(targets)
# test_targets = ternarize_tensor(test_targets)
# train_inputs_norm, test_inputs_norm = inputs, test_inputs

# For More detailed Discrete Model
# inputs, targets, test_inputs, test_targets = generate_random_split(seed=42, start = -7 * PI, end= 7 * PI, train_ratio=0.8, total_points=200)
# targets = torch.round(targets)
# test_targets = torch.round(test_targets)
# train_inputs_norm, test_inputs_norm = inputs, test_inputs

loss_histories = {}
weights = {}
test_losses = {}  # Dictionary to store test losses per seed

# Train the model for different seeds
#for i in [x for x in range(30, 51) if x != 32]:  # Adjust the range for more seeds

# Create a list to store the proportions for each neuron number
proportions = []
all_loss_records = [] 

# number of neurons starting from num_neuron = 2
for num_neuron in range(26, 29):  # 2, 3, ..., 7, for paper range(2,12)
    for num_hidden_layers in range(1, 11): # for paper range(1,6)
        test_losses.clear()  # Clear test losses for each new neuron configuration
        # for all weights, can be commented
        trained_models = {}
        for seed in range(0, 200):  # Adjust the range for more seeds
            print(f"Training with seed {seed}")
            loss_history, weight, trained_model = train_model(
            seed=seed, num_neuron=num_neuron, num_hidden_layers=num_hidden_layers, mode="decreasing", inputs= train_inputs_norm, 
            targets=targets, num_epochs=300, lr=0.002, print_every=100, w_0 =20)
            loss_histories[seed] = loss_history
            weights[seed] = weight
            # for all weights, can be commented
            trained_models[seed] = trained_model
        
            # Test after training
            test_loss = test_model(trained_model, test_inputs_norm, test_targets)
            test_losses[seed] = test_loss

        # Convert each value in the dictionary to a list
        # original weights
        # final_data_serializable = {k: v.tolist() for k, v in weights.items()}
        # for all weights, can be commented
        final_data_serializable = {}
        for seed, model in trained_models.items():
            seed_weights = {}
            seed_weights["input"] = model.linear.weight.detach().numpy().tolist()
            seed_weights["hidden"] = [hl.weight.detach().numpy().tolist() for hl in model.hidden_layers]
            final_data_serializable[seed] = seed_weights

        final_losses = {seed: history[-1] for seed, history in loss_histories.items()}
        sorted_loss_values = sorted(final_losses.values())
        min1 = sorted_loss_values[0]
        min2 = sorted_loss_values[1]
        min3 = sorted_loss_values[2]    

        # Check the global min of all seeds' optimization results
        min_loss = min(final_losses.values())

        # Check how many seeds' loss evidently larger than global min loss
        threshold = 1
        count_above_threshold = sum(
            1 for loss in final_losses.values() 
            if (loss - min_loss)/(min_loss + 1e-8) > threshold and (loss - min_loss) > 0.0001
        )

        # Calculate the probability of local min
        total_seeds = len(final_losses)
        proportion_above_threshold = count_above_threshold / total_seeds

        # === Add this for test loss local min analysis ===
        min_test_loss = min(test_losses.values())
        count_test_above_threshold = sum(
            1 for loss in test_losses.values() if (loss - min_test_loss)/(min_test_loss + 1e-8) > threshold and (loss - min_test_loss) > 0.0001
        )
        total_test_seeds = len(test_losses)
        proportion_test_above_threshold = count_test_above_threshold / total_test_seeds


        print(f"Test Loss Min: {min_test_loss:.6f}")
        print(f"Test Losses evidently above min: {count_test_above_threshold}/{total_test_seeds}")
        print(f"Test bad local min proportion: {proportion_test_above_threshold * 100:.4f}%")


        # === Analyze good-train AND good-test ===
        good_train_seeds = []
        good_train_and_test = []

        for seed in final_losses:
            train_loss = final_losses[seed]
            test_loss = test_losses[seed]

            is_good_train = (train_loss - min_loss) / (min_loss + 1e-8) <= threshold or (train_loss - min_loss) <= 0.0001
            is_good_test = (test_loss - min_test_loss) / (min_test_loss + 1e-8) <= threshold or (test_loss - min_test_loss) <= 0.0001

            if is_good_train:
                good_train_seeds.append(seed)
                if is_good_test:
                    good_train_and_test.append(seed)

        total_good_train = len(good_train_seeds)

        if total_good_train > 0:
            proportion_good_train_and_test = len(good_train_and_test) / total_good_train
            #print(f"Proportion of good-train AND good-test: {proportion_good_train_and_test * 100:.2f}%")
        else:
            proportion_good_train_and_test = None
            #print("No good-train seeds found, cannot compute proportion.")




        # Save the final dictionary to a text file
        with open(f'final_weights_N{num_neuron}_L{num_hidden_layers}.txt', 'w') as file:
            json.dump(final_data_serializable, file)

        print("Dictionary saved successfully.")
        print(f"Min Loss：{min_loss:.6f}")
        print(f"The number of local min evidently larger than global min：{count_above_threshold}/{total_seeds}")
        print(f"ratio：{proportion_above_threshold * 100:.4f}%")

        # Collect the proportion for the current neuron number
        proportions.append({
        "Neuron Number": num_neuron,
        "Layer Number": num_hidden_layers,
        "Min Loss": min_loss,
        "Second lowest min": min2,
        "Third lowest min": min3,
        "Proportion Above Threshold": proportion_above_threshold,
        "Min Test Loss": min_test_loss,
        "Test Proportion Above Threshold": proportion_test_above_threshold,
        "Good Train and Good Test":proportion_good_train_and_test
        })

        for seed in final_losses:
            all_loss_records.append([
                num_neuron,
                num_hidden_layers,
                seed,
                final_losses[seed],
                test_losses[seed],
            ])




        # Use the plotting function for training loss
        plot_loss_histories(
        loss_histories,
        filename=f"SinNN_Loss_Epoch_N{num_neuron}_L{num_hidden_layers}.png",
        xlabel='Epoch',
        ylabel='SinNN Loss'
        )

        # plot_loss_histories(
        # {seed: [test_losses[seed]] for seed in test_losses},
        # filename=f"Test_Losses_N{num_neuron}_L{num_hidden_layers}.png",
        # xlabel='Seed',
        # ylabel='Test Loss',
        # title=f'Test Losses (N={num_neuron}, L={num_hidden_layers})',
        # plot_type='bar'  
        # )



    # Save the proportions to a text file
with open('proportions_layerN28L10.txt', 'w') as f:
    for entry in proportions:
        f.write(f"Neuron number: {entry['Neuron Number']}, Layer number: {entry['Layer Number']}, "
                f"Global min: {entry['Min Loss']:.6f}, Second low min: {entry['Second lowest min']:.6f}, Third low min: {entry['Third lowest min']:.6f} bad local min propotion: {entry['Proportion Above Threshold']* 100:.4f}%, "
                f"Test min: {entry['Min Test Loss']:.6f}, bad test propotion: {entry['Test Proportion Above Threshold']* 100:.4f}%, "
                f"Good Train & Good Test: {entry['Good Train and Good Test']* 100:.6f}%\n")
        
with open('all_train_test_lossN28L10.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['num_neuron', 'num_hidden_layers', 'seed', 'train_loss', 'test_loss'])
    for record in all_loss_records:
        writer.writerow(record)

print("All seeds' train/test loss saved to all_train_test_loss.csv")


#plt.show()

