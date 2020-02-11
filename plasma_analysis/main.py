import cv2
import math
import csv
import ast
from utils import apply_threshold, abel_inversion, image_to_double, get_peak_to_valley, get_index_of_center_of_mass
from os import listdir
from os.path import join, isfile
from configparser import ConfigParser
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Read parameters and settings.
config = ConfigParser()
config.read('C:/Users/loasis/Desktop/python/config.ini')
parameters = config['parameters']
settings = config['settings']
directory_path = parameters['directory_path']
scan_parameter = parameters['scan_parameter']
scan_parameter_alias = parameters['scan_parameter_alias']
y_cal = float(parameters['micron_per_pixel'])
wavelength = int(parameters['wavelength'])
left_pixel = int(parameters['left_pixel'])
save_video = ast.literal_eval(settings['save_video'])
save_image = ast.literal_eval(settings['save_image'])
negate_density_map = ast.literal_eval(settings['negate_density_map'])
average_lineouts = ast.literal_eval(settings['average_lineouts'])

print('Loading data...')

# Load Phasics files.
all_files = listdir(directory_path)
phase_files = [f for f in all_files if isfile(join(directory_path, f)) and f.startswith('PHA')]
raw_files = [f for f in all_files if isfile(join(directory_path, f)) and f.startswith('RAW')]
phase_files.sort()
raw_files.sort()

# Load scan data.
parent_directory_path = directory_path[:-26]
scan_file = join(parent_directory_path, [f for f in listdir(parent_directory_path) if f.startswith('ScanData')][0])
scan_number = scan_file[-6:-4]
scan_data = np.genfromtxt(scan_file, skip_header=1, missing_values='')
with open(scan_file, 'r') as f:
    d_reader = csv.DictReader(f, delimiter='\t')
    headers = d_reader.fieldnames
parameter_index = headers.index(scan_parameter)
parameter_values = scan_data[:, parameter_index]

# Determine the step length used in the scan.
differences = [t - s for s, t in zip(parameter_values, parameter_values[1:])]
differences = np.asarray(differences)
step_lengths = differences[differences > 0]
average_step_length = np.mean(step_lengths)
step_lengths = step_lengths[step_lengths > 0.9*average_step_length]
average_step_length = np.mean(step_lengths)

# Determine which indices correspond to a change in the value of the scan parameter (first shots).
first_shots = np.argwhere(differences > 0.9*average_step_length)
first_shots = np.insert(first_shots+1, 0, 0)

# Select ROI.
first_file = join(directory_path, phase_files[0])
last_file = join(directory_path, phase_files[-1])
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(cv2.imread(first_file, cv2.IMREAD_GRAYSCALE), aspect='equal', cmap='jet')
ax2.imshow(cv2.imread(last_file, cv2.IMREAD_GRAYSCALE), aspect='equal', cmap='jet')
ax1.set_title('First shot')
ax2.set_title('Last shot')
fig.suptitle('Select ROI in both panels by clicking at two diagonally opposite points')
for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
p1, p2, p3, p4 = plt.ginput(4)
plt.close(fig)
x_min = int(min([p1[0], p2[0], p3[0], p4[0]]))
x_max = int(max([p1[0], p2[0], p3[0], p4[0]]))
y_min = int(min([p1[1], p2[1], p3[1], p4[1]]))
y_max = int(max([p1[1], p2[1], p3[1], p4[1]]))

# Initialize data structures.
n_columns = x_max-x_min
n_parameter_values = len(parameter_values)
n_phase_files = len(phase_files)
density_lineouts = np.zeros((n_parameter_values, n_columns))
phase_lineouts = np.zeros((n_parameter_values, n_columns))
density_maps = []
phase_maps = []
center_of_mass_indices = []

print('Plotting lineouts...')

# Main loop.
density_max = 0
i_phase_file = 0
for i_parameter_value in range(n_parameter_values):
    phase_file = join(directory_path, phase_files[i_phase_file])
    shot_number = int(phase_file[-7:-4])
    if shot_number == i_parameter_value+1:
        # Read image.
        raw_image = cv2.imread(phase_file, cv2.IMREAD_GRAYSCALE)
        PtV = get_peak_to_valley(join(directory_path, raw_files[i_phase_file]))
        phase_map = image_to_double(raw_image)*(2*math.pi*PtV)
        phase_maps.append(phase_map)
        # Get density outline.
        cropped_phase_map = phase_map[y_min:y_max, x_min:x_max]
        thresholded_phase_map = apply_threshold(cropped_phase_map, 0.5)
        half_density_map, density_map = abel_inversion(cropped_phase_map, thresholded_phase_map, y_cal, wavelength)
        if negate_density_map:
            half_density_map = (-1)*half_density_map
            density_map = (-1)*density_map
        x = np.amax(density_map)
        if x > density_max:
            density_max = x
        density_maps.append(density_map)
        density_lineouts[i_parameter_value, :] = half_density_map[-2, :]
        # Get phase outline.
        i = get_index_of_center_of_mass(thresholded_phase_map)
        center_of_mass_indices.append(i)
        phase_lineouts[i_parameter_value, :] = cropped_phase_map[i, :]
        i_phase_file += 1
    if i_phase_file > n_phase_files:
        break
density_maps = density_maps[::-1]
np.save('C:/Users/loasis/Desktop/python/density_lineouts.npy', density_lineouts)
np.save('C:/Users/loasis/Desktop/python/phase_lineouts.npy', phase_lineouts)

# If there are multiple shots for each parameter value, take either the average of the shots or simply the first shot.
if average_lineouts:
    n = len(first_shots)
    final_density_lineouts = np.zeros((n, n_columns))
    final_phase_lineouts = np.zeros((n, n_columns))
    for i in range(n-1):
        j = first_shots[i]
        jj = first_shots[i+1]
        temp1 = density_lineouts[j:jj, :]
        temp2 = phase_lineouts[j:jj, :]
        all_zeros = not np.any(temp1) 
        if all_zeros:
            # Avoid taking mean of empty array.
            final_density_lineouts[i, :].fill(np.nan)
            final_phase_lineouts[i, :].fill(np.nan)
        else:
            final_density_lineouts[i, :] = np.mean(temp1[np.any(temp1, axis=1)], axis=0)
            final_phase_lineouts[i, :] = np.mean(temp2[np.any(temp2, axis=1)], axis=0)
    j = first_shots[-1]
    temp1 = density_lineouts[j:, :]
    temp2 = phase_lineouts[j:, :]
    final_density_lineouts[-1, :] = np.mean(temp1[np.any(temp1, axis=1)], axis=0)
    final_phase_lineouts[-1, :] = np.mean(temp2[np.any(temp2, axis=1)], axis=0)
else:
    final_density_lineouts = density_lineouts[first_shots, :]
    final_phase_lineouts = phase_lineouts[first_shots, :]

# Calculate the µm range in z-direction (laser direction).
pixels = list(range(1, 1600, 4))
i = 0
pixel_shift = 0
while pixels[i] <= left_pixel:
    if pixels[i+1] > left_pixel:
        pixel_shift = i
        break
    i += 1
z = list(range(x_min, x_max))
z = [y_cal*(n+pixel_shift) for n in z]
np.save('C:/Users/loasis/Desktop/python/z.npy', z)

# Plot and save the result.
fig, (ax1, ax2) = plt.subplots(1, 2)
im1 = ax1.imshow(final_density_lineouts, aspect='auto', cmap='jet',
                 extent=[z[0], z[-1], parameter_values[-1], parameter_values[0]])
ax1.set_title('Density lineouts')
cbar1 = fig.colorbar(im1, ax=ax1)
cbar1.set_label('Density [$10^{18}$ e$^{-}$/cm$^{3}$]')
im2 = ax2.imshow(final_phase_lineouts, aspect='auto', cmap='jet',
                 extent=[z[0], z[-1], parameter_values[-1], parameter_values[0]])
ax2.set_title('Phase lineouts')
cbar2 = fig.colorbar(im2, ax=ax2)
cbar2.set_label('Phase shift [rad]')
for ax in [ax1, ax2]:
    ax.set_xlabel('z [µm]')
    ax.set_ylabel('Position of ' + scan_parameter_alias + ' [mm]')
fig.set_size_inches(16, 8)
plt.suptitle('Scan ' + str(scan_number))
if save_image:
    fig.savefig('C:/Users/loasis/Desktop/python/scan' + str(scan_number) + '.png')
plt.show(block=True)

# Plot how well the center of mass indices are determined
n_rows = 5
n_columns = 4
n_checks = n_rows*n_columns
i_selected_files = [int(i) for i in np.linspace(0, n_phase_files, n_checks, endpoint=False)]
_, axes = plt.subplots(n_rows, n_columns)
for ax, i in zip(axes.ravel(), i_selected_files):
    ax.imshow(phase_maps[i])
    roi = matplotlib.patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
    ax.add_patch(roi)
    line = matplotlib.lines.Line2D([x_min, x_max], [y_min+center_of_mass_indices[i], y_min+center_of_mass_indices[i]], linewidth=1, color='r')
    ax.add_line(line)
    ax.set_title('Shot #' + str(i+1))
plt.suptitle('ROI and center of mass for some shots... Does it look correct?')
plt.show(block=True)

# Create and save density map animation.
if save_video:
    fig, ax = plt.subplots(1, 1)
    image = ax.imshow(density_maps[0], aspect='equal', vmin=0, vmax=density_max, cmap='jet')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    def animate(frame):
        global density_maps, image
        image.set_array(density_maps[frame])
        return image

    total_number_of_frames = len(density_maps)
    animation = FuncAnimation(fig, animate, total_number_of_frames, fargs=[], interval=1000/10)
    animation.save('C:/Users/loasis/Desktop/python/scan' + str(scan_number) + '.mp4')
