import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.cm as cm
import random

pi = np.pi
sin = np.sin
sign = np.sign

# Create a meshgrid for x and y
w1 = np.linspace(-2.1, 2.1, 400)
w2 = np.linspace(-2.1, 2.1, 400)
w1, w2 = np.meshgrid(w1, w2)


# def y(w1, w2, x):
#     return sin(w1 * x) + sin(w2 * x)

def y_c(w1, w2, x):
    return sin(w1 * x) + sin(w2 * x)

def y(w1, w2, x):
    return sign(sign(sin(sign(w1) * x)) + sign(sin(sign(w2) * x)))


def half_binarzed_y(w1, w2, x):
    return sign(sign(sin(w1 * x)) + sign(sin(w2 * x)))


# Calculate z based on the given function
z = (
        (y_c(w1, w2, -3 / 2 * pi) - 2) ** 2
        + (y_c(w1, w2, - pi / 2) + 2) ** 2
        + (y_c(w1, w2, pi / 2) - 2) ** 2
        + (y_c(w1, w2, 3 / 2 * pi) + 2) ** 2
     ) / 4


with open('final_weights.txt', 'r') as file:
    loaded_data = json.load(file)

# Convert lists back to NumPy arrays if necessary
loaded_data_arrays = {k: np.array(v) for k, v in loaded_data.items()}

# Ensure w1, w2, z, and loaded_data_arrays are defined

fig, ax = plt.subplots(figsize=(20, 12))
levels = np.linspace(0, 12, 20)
contour = ax.contourf(w1, w2, z, levels=levels, cmap='viridis')
contour_lines = ax.contour(w1, w2, z, levels=levels, colors='black', linewidths=0.5)

# Adding the colorbar and setting tick labels correctly
colorbar = plt.colorbar(contour)
# Reducing the number of grid lines (ticks) on the colorbar
colorbar.set_ticks([0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12])  # Set fewer tick positions
colorbar.ax.tick_params(labelsize=28)  # Change tick label size

# Labels and title
plt.xlabel('$w_1$', fontsize=34)
plt.ylabel('$w_2$', fontsize=34)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Generating a colormap for the different keys (weights)
colors = cm.rainbow(np.linspace(1, 0, 80))

# Define possible markers
markers = ['o']  # 'x' for cross, '^' for triangle, 's' for square
label_added = False  # Flag to add label only once

# Add crosses for each weight with a different color
for i, (key, value) in enumerate(loaded_data_arrays.items()):
    weight1 = value[0][0]  # First weight
    weight2 = value[1][0]  # Second weight
    marker = random.choice(markers)
    
    # Scatter with unique label only once
    if marker == 'o' and not label_added:
        ax.scatter(weight1, weight2, color=colors[1], marker=marker, s=200, label='Classical')
        label_added = True
    else:
        ax.scatter(weight1, weight2, color=colors[1], marker=marker, s=200)  # No label for subsequent points

# Add a Quantum point marker
ax.scatter(1, 1, color='y', marker='*', s=300, label='Quantum')

# Add a legend above the plot
plt.legend(bbox_to_anchor=(0.5, 1.10), loc='center', fontsize=36, ncol=2, borderaxespad=0.)

# Adjust the layout to prevent cutting off the legend
plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)

# Save the plot to a file with a valid path
plt.savefig('ContinuousResults.png', dpi=400, bbox_inches='tight')

# Show the plot
plt.show()

