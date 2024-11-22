import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.cm as cm
import random
import matplotlib.patheffects as patheffects

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
        (y(w1, w2, -3 / 2 * pi) - 1) ** 2
        + (y(w1, w2, - pi / 2) + 1) ** 2
        + (y(w1, w2, pi / 2) - 1) ** 2
        + (y(w1, w2, 3 / 2 * pi) + 1) ** 2
     ) / 4


with open('final_Bweights.txt', 'r') as file:
    loaded_data = json.load(file)

# Convert lists back to NumPy arrays if necessary
loaded_data_arrays = {k: np.array(v) for k, v in loaded_data.items()}

# Create the plot
fig, ax = plt.subplots(figsize=(20, 12))
# Enable LaTeX font rendering in Matplotlib
plt.rcParams.update({
    'text.usetex': True,             # Use LaTeX for all text rendering
    'font.family': 'serif',          # Set the font family to serif
    'font.serif': ['Computer Modern'], # Explicitly use Computer Modern
})
levels = np.linspace(0, 4.5, 8)
contour = ax.contourf(w1, w2, z, levels=levels, cmap='viridis')
contour_lines = ax.contour(w1, w2, z, levels=levels,
                           colors='black', linewidths=0.5)
# ax.clabel(contour_lines, inline=True, fontsize=20)

# Adding the colorbar and setting tick labels correctly
colorbar = plt.colorbar(contour)
# Reducing the number of grid lines (ticks) on the colorbar
colorbar.set_ticks([0, 1.5, 3, 4.5])  # Set fewer tick positions
colorbar.ax.tick_params(labelsize=28)  # Change tick label size
# Iterate over each tick label on the colorbar axis and apply path effects for a bolder look
for tick_label in colorbar.ax.get_yticklabels():
    tick_label.set_path_effects([patheffects.withStroke(linewidth=1.25, foreground='black')])





# Labels with LaTeX-style rendering
plt.xlabel('$w_1$', fontsize=50, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])  # LaTeX with italic formatting for w_1
plt.ylabel('$w_2$', fontsize=50, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])  # LaTeX with italic formatting for w_2

plt.xticks(fontsize=32, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
plt.yticks(fontsize=32, path_effects=[patheffects.withStroke(linewidth=1.25, foreground='black')])
# plt.title('Contour plot of MSE with 6 inputs and 2 weights.  ')
# plt.figtext(0.5, 0.01,
#             'y_target(t) = 2 * sin(t).  '
#             'y_pred(t) = sin(W_1 * t) + sin(W_2 * t).  '
#             't = -3/2*pi, -pi, -pi/2, pi/2, pi, 3/2*pi.',
#             ha="center", fontsize=10,
#             bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

# Generate a colormap for the different keys (weights)
#colors = cm.rainbow(np.linspace(0, 1, len(loaded_data_arrays)))
colors = cm.rainbow(np.linspace(1, 0, 100))

# Define possible markers
markers = ['o']  # 'x' for cross, '^' for triangle, 's' for square
# Add crosses for each weight with a different color
label_added = False  # Flag to add label only once
for i, (key, value) in enumerate(loaded_data_arrays.items()):
    weight1 = value[0][0]  # First weight
    weight2 = value[1][0]  # Second weight
    marker = random.choice(markers)
    print(marker)
    if marker == 'o':
        if not label_added:
            ax.scatter(weight1, weight2, color=colors[1], marker=marker, s=200, label=r'$\textbf{Classical}$')
            label_added = True
        else:
            ax.scatter(weight1, weight2, color=colors[1], marker=marker, s=200)  # No label for subsequent points
    else:
        ax.scatter(weight1, weight2, color=colors[1], marker=marker, s=200, label=f'Seed {key}')

    


# Optionally, add a legend to indicate which color corresponds to which key
ax.scatter(1, 1, color='y', marker='*', s=1000, label=r'$\textbf{Quantum}$')


plt.legend(
    bbox_to_anchor=(0.5, 1.10),  # Positioning the legend above the plot
    loc='center',                # Center it horizontally
    fontsize=46,                 # Set the font size
    ncol=2,                      # Number of columns
    borderaxespad=0.,            # Adjust padding
    frameon=False,               # Disable the border
    handletextpad=0.01            # Reduce spacing between marker and label
)

# Adjust the layout to prevent cutting off the legend
plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)
#plt.subplots_adjust(right=0.90)
#plt.subplots_adjust(bottom=0.20)  # Increase the bottom margin

#plt.legend(loc='upper right', fontsize=12)
# Save the plot to a file
plt.savefig('contour_plot_MSE_binary.png', dpi=400, bbox_inches='tight')

# Show the plot
plt.show()
