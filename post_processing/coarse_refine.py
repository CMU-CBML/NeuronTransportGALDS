import numpy as np

def calculate_radius(parent_coordinates):
    center_coord = np.mean(parent_coordinates, axis=0)
    radius = np.max(np.linalg.norm(parent_coordinates - center_coord, axis=1))
    return radius

def create_new_coordinates(parent_surface_points, shift_vector, radius):
    original_raidus = calculate_radius(parent_surface_points)
    size_factor = radius / original_raidus
    new_coordinates = parent_surface_points + shift_vector
    center_point = new_coordinates[0]
    point_bias = new_coordinates - center_point
    new_coordinates = center_point + point_bias * size_factor
    return new_coordinates

def coarse_refine_create_points(parent_id, child_id, coordinates, point_data, refine_num=100, inverse_interpolation=False):
    parent_id = int(parent_id)
    child_id = int(child_id)
    parent_coordinate = coordinates[parent_id]
    child_coordinate = coordinates[child_id]
    # calculate radius for parent and child
    parent_radius = calculate_radius(coordinates[parent_id:parent_id + 17])
    child_radius = calculate_radius(coordinates[child_id:child_id + 17])
    # interplate the radius
    radius_steps = (child_radius - parent_radius) / (refine_num + 1)
    # calculate the vector from parent to child
    vector = child_coordinate - parent_coordinate
    vector_steps = vector / (refine_num + 1)
    append_coordinates = np.zeros((1,3))
    new_point_data_in_element_ls = []
    for i in range(refine_num):
        # first get the parent surface points (idx parent_id + 0 - 16)
        parent_idxs = np.arange(parent_id, parent_id + 17)
        child_idxs = np.arange(child_id, child_id + 17)
        # calculate the raidus for interpolation
        radius_on_surface = parent_radius + i * radius_steps
        # get the coordinates of the parent surface points
        parent_surface_points = coordinates[parent_idxs] 
        # calculate the new point, which is the parent point + i * vector_steps
        # new_points = parent_surface_points + i * vector_steps
        if inverse_interpolation is False:
            new_points = create_new_coordinates(parent_surface_points, (i+1) * vector_steps, radius_on_surface)
        else:
            child_surface_points = coordinates[child_idxs]
            new_points = create_new_coordinates(child_surface_points, -(refine_num + 0 - i) * vector_steps, radius_on_surface)
        # append the new point to the coordinates
        append_coordinates = np.vstack((append_coordinates, new_points))
        # linear interpolation of the point data
        new_point_data_in_element = {}
        for key_name in point_data:
            parent_data = point_data[key_name][parent_idxs]
            child_data = point_data[key_name][child_idxs]
            new_data = parent_data + i * (child_data - parent_data) / refine_num
            new_point_data_in_element[key_name] = new_data
        new_point_data_in_element_ls.append(new_point_data_in_element)
    # Assemble all the new_point_data into same dict
    new_point_data = {}
    for key_name in point_data:
        new_point_data[key_name] = np.concatenate([data[key_name] for data in new_point_data_in_element_ls], axis=0)
    # Concatenate the new point data with the original point data
    new_total_point_data = {}
    for key_name in point_data:
        new_total_point_data[key_name] = np.concatenate((point_data[key_name], new_point_data[key_name]), axis=0)
    
    append_coordinates = np.delete(append_coordinates, 0, axis=0)  # Remove the initial zero row
    return append_coordinates, new_total_point_data

