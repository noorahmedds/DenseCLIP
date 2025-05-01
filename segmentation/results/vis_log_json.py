import json

import matplotlib.pyplot as plt

# Path to the JSON log file
# log_file_path = '/home/noah00001/Desktop/projects/throwaway/DenseCLIP/segmentation/results/ft_sdc_pin_bicycle/20250427_185730.log.json'
log_file_path = '/home/noah00001/Desktop/projects/throwaway/DenseCLIP/segmentation/results/ft_sdc_pin/20250427_195255.log.json'

# Load the JSON log data
with open(log_file_path, 'r') as f:
    log_data = [json.loads(line) for line in f]

# Extract train and validation loss
train_iterations = []
train_losses = []
val_iterations = []
val_losses = []

for ix, entry in enumerate(log_data):
    if 'mode' in entry and entry['mode'] == 'train':
        train_iterations.append(entry['iter'])
        train_losses.append(entry['loss'])
        if "loss_val" in entry:
            val_iterations.append(entry['iter'])
            val_losses.append(entry['loss_val'])

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_iterations, train_losses, label='Train Loss', marker='o', linestyle='-')
plt.plot(val_iterations, val_losses, label='Validation Loss', marker='x', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()