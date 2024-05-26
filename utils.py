import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Globals
SCALING_FACTOR = 54.67


# OVERALL ---------------------------------------
def load_data(filename):
    out = {}
    with open(filename, 'r') as file:
        for line in file:
            name, x, y = line.split()
            out[name] = (float(x), float(y))
    return out

def load_dataset(pv_img, tv_img):
    validation_pts_tv = load_data(f'data/annotations/{tv_img}_blue.txt')
    validation_pts_pv = load_data(f'data/annotations/{pv_img}_blue.txt')
    reference_pts_tv = load_data(f'data/annotations/{tv_img}_white.txt')
    reference_pts_pv = load_data(f'data/annotations/{pv_img}_white.txt')
    # Slice TV points 
    validation_pts_tv = {k: validation_pts_tv[k] for k in validation_pts_pv}
    reference_pts_tv = {k: reference_pts_tv[k] for k in reference_pts_pv}
    return reference_pts_pv, reference_pts_tv, validation_pts_pv, validation_pts_tv

def get_images(pv_img, tv_img):
    pv_name = f'data/images/{pv_img}.JPG'
    tv_name = f'data/images/{tv_img}.JPG'
    return cv2.cvtColor(cv2.imread(pv_name), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(tv_name), cv2.COLOR_BGR2RGB)
    
def to_array(l):
    return np.array([[l[d][0], l[d][1]] for d in l])

def get_numpy_arr(pts):
    pts_arr = to_array(pts)
    homogeneous_c = np.ones((pts_arr.shape[0], 1))
    return np.hstack((pts_arr, homogeneous_c))

# Transform list to homogeneous numpy array
def transform_points(reference_pts_pv, reference_pts_tv, validation_pts_pv, validation_pts_tv):
    reference_pts_pv_arr = get_numpy_arr(reference_pts_pv)
    reference_pts_tv_arr = get_numpy_arr(reference_pts_tv)
    validation_pts_pv_arr = get_numpy_arr(validation_pts_pv)
    validation_pts_tv_arr = get_numpy_arr(validation_pts_tv)
    return reference_pts_pv_arr, reference_pts_tv_arr, validation_pts_pv_arr, validation_pts_tv_arr

# Transform reference points to Top View
def predict(pts, h):
    pred_pts = h@pts.T
    return (pred_pts / pred_pts[2]).T
    
def plot_setup(img_tv, img_pv, reference_result_dict, validation_result_dict):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Reference- and Validation Points w Pixel Density')
    axs[1].imshow(img_tv)
    axs[0].imshow(img_pv)
    axs[0].axis('off')
    axs[1].axis('off')
    # Reference Points
    x_offset = -300
    y_offset = +200
    for name, pt in reference_result_dict.items():
        axs[0].plot(pt['coordinates_pv'][0], pt['coordinates_pv'][1], marker='X', color='whitesmoke', markersize=8)
        axs[0].annotate(f'{pt["pixel_density"]:.2f}', 
                        xy=(pt['coordinates_pv'][0]+x_offset, pt['coordinates_pv'][1]+y_offset),
                        bbox=dict(facecolor='whitesmoke', edgecolor='white', boxstyle="round", alpha=0.8))
        axs[1].plot(pt['coordinates_tv'][0], pt['coordinates_tv'][1], marker='X', color='whitesmoke', markersize=8)
        
    # Validation Points
    for name, pt in validation_result_dict.items():
        axs[0].plot(pt['coordinates_pv'][0], pt['coordinates_pv'][1], marker='X', color='steelblue', markersize=8)
        axs[0].annotate(f'{pt["pixel_density"]:.2f}', 
                        xy=(pt['coordinates_pv'][0]+x_offset, pt['coordinates_pv'][1]+y_offset),
                        bbox=dict(facecolor='steelblue', edgecolor='royalblue', boxstyle="round", alpha=0.8),
                        color='white')
        axs[1].plot(pt['coordinates_tv'][0], pt['coordinates_tv'][1], marker='X', color='steelblue', markersize=8)  
    plt.subplots_adjust(bottom=0, wspace=0)
    plt.show()

def plot_predictions(img_tv, img_pv, reference_result_dict, validation_result_dict, fname=False):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # fig.suptitle('Reference- and Validation Points, their Predictions, and Error')
    ax.imshow(img_tv)
    ax.axis('off')
    x_offset = -150
    y_offset = +200
    # Reference Points
    for name, pt in reference_result_dict.items():
        ax.plot(pt['coordinates_tv'][0], pt['coordinates_tv'][1], marker='X', color='whitesmoke', markersize=8)
        ax.plot(pt['predicted_coordinates_tv'][0], pt['predicted_coordinates_tv'][1], marker='X', color='lightsalmon', markersize=8)
        ax.annotate(f'{pt["error"]:.2f}', 
                    xy=(pt['coordinates_tv'][0]+x_offset, pt['coordinates_tv'][1]+y_offset),
                    bbox=dict(facecolor='whitesmoke', edgecolor='white', boxstyle="round", alpha=0.8))
        
    # Validation Points
    errors = []
    for name, pt in validation_result_dict.items():
        errors.append(pt["error"])
        ax.plot(pt['coordinates_tv'][0], pt['coordinates_tv'][1], marker='X', color='steelblue', markersize=8)  
        ax.plot(pt['predicted_coordinates_tv'][0], pt['predicted_coordinates_tv'][1], marker='X', color='cyan', markersize=8)
        ax.annotate(f'{pt["error"]:.2f}', 
                    xy=(pt['coordinates_tv'][0]+x_offset, pt['coordinates_tv'][1]+y_offset),
                    bbox=dict(facecolor='steelblue', edgecolor='royalblue', boxstyle="round", alpha=0.8),
                    color='white')

    # Mean Error:
    ax.annotate(f'{np.mean(errors) / SCALING_FACTOR * 100 :.2f} cm', xy=(100, 150), color='white',
        bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', alpha=0.8))
    # if name:
        # plt.savefig(f'predictions_{fname}.png', dpi=300)
    return ax


# Calculate Errors
def calculate_error(pred_pts, gt_pts):
    return np.mean([np.linalg.norm(pred-gt) for pred, gt in zip(pred_pts[:, :2], gt_pts[:, :2])])

def print_errors(predicted_reference_pts_tv, reference_pts_tv_arr, predicted_validation_pts_tv, validation_pts_tv_arr):
    print(f'Reference  Error: {calculate_error(predicted_reference_pts_tv, reference_pts_tv_arr) :.2f} PX!')
    print(f'Validation Error: {calculate_error(predicted_validation_pts_tv, validation_pts_tv_arr):.2f} PX!')
    print('---')
    print(f'Reference  Error: {calculate_error(predicted_reference_pts_tv, reference_pts_tv_arr) / SCALING_FACTOR * 100 :.2f} CM!')
    print(f'Validation Error: {calculate_error(predicted_validation_pts_tv, validation_pts_tv_arr)/ SCALING_FACTOR * 100 :.2f} CM!')

def calculate_errors(pred_pts, gt_pts):
    return [np.linalg.norm(pred-gt) for pred, gt in zip(pred_pts[:, :2], gt_pts[:, :2])]

def calc_distance(a, b):
    return sum((y-x)**2 for x, y in zip(a, b)) ** 0.5

def get_nearest_point_distance(point, point_list):
    distances = [calc_distance(point, pt) for pt in point_list]
    return min(distances)

def get_result_dict(point_dict_pv, point_arr_tv, predicted_points_tv, errors, densities, reference_pts_pv):
    error_dict = {}
    for i, (name, coord) in enumerate(point_dict_pv.items()):
        error_dict[name] = {}
        error_dict[name]['coordinates_pv'] = coord
        error_dict[name]['coordinates_tv'] = (point_arr_tv[i][0], point_arr_tv[i][1])
        error_dict[name]['predicted_coordinates_tv'] = (predicted_points_tv[i][0], predicted_points_tv[i][1])
        error_dict[name]['error'] = errors[i]
        error_dict[name]['pixel_density'] = densities[i]
        error_dict[name]['distance_to_next_ref_pt'] = get_nearest_point_distance(coord, reference_pts_pv)
    return error_dict

# Get Point Density
def get_surrounding_points(point):
    point_right, point_left, point_up, point_down = point.copy(), point.copy(), point.copy(), point.copy()
    point_right[0] += 2*SCALING_FACTOR
    point_left[0] -= 2*SCALING_FACTOR
    point_up[1] += 2*SCALING_FACTOR
    point_down[1] -= 2*SCALING_FACTOR
    return point_right, point_left, point_up, point_down

def get_point_density(point, h_inv):
    point_right, point_left, point_up, point_down = get_surrounding_points(point)
    # Predict
    points = np.vstack((point, point_right, point_left, point_up, point_down))
    predicted_points = predict(points, h_inv)
    dists = []
    for surrounding_point in predicted_points[1:]:
        dists.append(np.linalg.norm(predicted_points[0,:2] - surrounding_point[:2]))
    return np.mean(dists)

def get_densities(points, h_inv):
    densities = []
    for pt in points: 
        densities.append(get_point_density(pt, h_inv))
    return densities



# NOISED ----------------------
def add_noise(points, m=0, std=1):
    zeros = np.zeros((points.shape[0], 1))
    noise = np.random.normal(m, std, points[:,:-1].shape)
    noise = np.hstack((noise, zeros))
    return points + noise

def get_result_dict_noised(point_dict_pv, point_arr_tv, predicted_points_tv, errors, densities, point_arr_pv_noised, point_arr_tv_noised):
    error_dict = {}
    for i, (name, coord) in enumerate(point_dict_pv.items()):
        error_dict[name] = {}
        error_dict[name]['coordinates_pv'] = coord
        error_dict[name]['coordinates_tv'] = (point_arr_tv[i][0], point_arr_tv[i][1])
        error_dict[name]['coordinates_pv_noised'] = (point_arr_pv_noised[i][0], point_arr_pv_noised[i][1])
        error_dict[name]['coordinates_tv_noised'] = (point_arr_tv_noised[i][0], point_arr_tv_noised[i][1])
        error_dict[name]['predicted_coordinates_tv'] = (predicted_points_tv[i][0], predicted_points_tv[i][1])
        error_dict[name]['error'] = errors[i]
        error_dict[name]['pixel_density'] = densities[i]
    return error_dict

def plot_setup_noised(img_tv, img_pv, reference_result_dict, validation_result_dict):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(bottom=0, wspace=0)
    axs[0].imshow(img_pv)
    axs[1].imshow(img_tv)
    axs[0].axis('off')
    axs[1].axis('off')
    for name, pt in reference_result_dict.items():          
        axs[0].plot(pt['coordinates_pv'][0], pt['coordinates_pv'][1], marker='X', color='whitesmoke', markersize=8)
        axs[0].plot(pt['coordinates_pv_noised'][0], pt['coordinates_pv_noised'][1], marker='X', color='tomato', markersize=8)
        axs[0].annotate("", xy=(pt['coordinates_pv_noised'][0], pt['coordinates_pv_noised'][1]), 
                    xytext=(pt['coordinates_pv'][0], pt['coordinates_pv'][1]),
                    #arrowprops=dict(facecolor='black'))
                    arrowprops=dict(arrowstyle="simple", facecolor='black'))
        axs[1].plot(pt['coordinates_tv'][0], pt['coordinates_tv'][1], marker='X', color='whitesmoke', markersize=8)
        axs[1].plot(pt['coordinates_tv_noised'][0], pt['coordinates_tv_noised'][1], marker='X', color='tomato', markersize=8)
        axs[1].annotate("", xy=(pt['coordinates_tv_noised'][0], pt['coordinates_tv_noised'][1]), 
                    xytext=(pt['coordinates_tv'][0], pt['coordinates_tv'][1]),
                    #arrowprops=dict(facecolor='black'))
                    arrowprops=dict(arrowstyle="simple", facecolor='black'))
    # plt.savefig('setup_noised.png', dpi=300)


# OUTLIER -----------
def add_outlier(points, i, magnitude):
    points_ = points.copy()
    noise = np.random.normal(0, 10, 2)*magnitude
    points_[i, :2] = points_[i, :2] + noise 
    return points_# + noise
