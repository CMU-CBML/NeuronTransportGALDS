import numpy as np

def create_cell_p2p(parent_idx, child_idx, coordinates):
    parent_idx = int(parent_idx)
    child_idx = int(child_idx)
    parent_idx2_coord = coordinates[parent_idx + 2]
    child_idx2_coord = coordinates[child_idx + 2]
    child_idx1_coord = coordinates[child_idx + 1]
    child_idx11_coord = coordinates[child_idx + 11]
    child_idx3_coord = coordinates[child_idx + 3]
    distance_parent2_child2 = np.linalg.norm(parent_idx2_coord - child_idx2_coord)
    distance_parent2_child1 = np.linalg.norm(parent_idx2_coord - child_idx1_coord)
    distance_parent2_child11 = np.linalg.norm(parent_idx2_coord - child_idx11_coord)
    distance_parent2_child3 = np.linalg.norm(parent_idx2_coord - child_idx3_coord)
    distance_list = [distance_parent2_child2, distance_parent2_child1, distance_parent2_child11, distance_parent2_child3]
    # get the index of the minimum value
    min_idx = np.argmin(distance_list)

    if min_idx == 0:
        # parent 2 is closest to child 2
        parent_idx1_coord = coordinates[parent_idx + 1]
        distance_parent1_child1 = np.linalg.norm(parent_idx1_coord - child_idx1_coord)
        distance_parent1_child3 = np.linalg.norm(parent_idx1_coord - child_idx3_coord)
        if distance_parent1_child1 < distance_parent1_child3:
            # parent 1 is closest to child 1
            cell_1_bias_parent = cell_1_bias_child = [10, 5, 0, 7]
            cell_2_bias_parent = cell_2_bias_child = [5,9,4,0]
            cell_3_bias_parent = cell_3_bias_child = [0, 4, 15, 12]
            cell_4_bias_parent = cell_4_bias_child = [7, 0, 12, 16]
            cell_5_bias_parent = cell_5_bias_child = [8, 10, 7, 3]
            cell_6_bias_parent = cell_6_bias_child = [8, 2, 5, 10]
            cell_7_bias_parent = cell_7_bias_child = [2, 6, 9, 5]
            cell_8_bias_parent = cell_8_bias_child = [9, 6, 1, 4]
            cell_9_bias_parent = cell_9_bias_child = [4, 1, 13, 15]
            cell_10_bias_parent = cell_10_bias_child = [12, 15, 13, 11]
            cell_11_bias_parent = cell_11_bias_child = [16, 12, 11, 14]
            cell_12_bias_parent = cell_12_bias_child = [3, 7, 16, 14]
        else:
            cell_1_bias_parent = [10, 5, 0, 7]
            cell_2_bias_parent = [5,9,4,0]
            cell_3_bias_parent = [0, 4, 15, 12]
            cell_4_bias_parent = [7, 0, 12, 16]
            cell_5_bias_parent = [8, 10, 7, 3]
            cell_6_bias_parent = [8, 2, 5, 10]
            cell_7_bias_parent = [2, 6, 9, 5]
            cell_8_bias_parent = [9, 6, 1, 4]
            cell_9_bias_parent  = [4, 1, 13, 15]
            cell_10_bias_parent = [12, 15, 13, 11]
            cell_11_bias_parent = [16, 12, 11, 14]
            cell_12_bias_parent = [3, 7, 16, 14]

            cell_1_bias_child = [9, 5, 0, 4]
            cell_2_bias_child = [5, 10, 7, 0]
            cell_3_bias_child = [0, 7, 16, 12]
            cell_4_bias_child = [4, 0, 12, 15]
            cell_5_bias_child = [6, 9, 4, 1]
            cell_6_bias_child = [6, 2, 5, 9]
            cell_7_bias_child = [2, 8, 10, 5]
            cell_8_bias_child = [10, 8, 3, 7]
            cell_9_bias_child = [7, 3, 14, 16]
            cell_10_bias_child = [12, 16, 14, 11]
            cell_11_bias_child = [15, 12, 11, 13]
            cell_12_bias_child = [1, 4, 15, 13]

    elif min_idx == 1:
        # parent 2 is closest to child 1
        parent_idx1_coord = coordinates[parent_idx + 1]
        distance_parent1_child11 = np.linalg.norm(parent_idx1_coord - child_idx11_coord)
        distance_parent1_child2 = np.linalg.norm(parent_idx1_coord - child_idx2_coord)
        cell_1_bias_parent = [10, 5, 0, 7]
        cell_2_bias_parent = [5,9,4,0]
        cell_3_bias_parent = [0, 4, 15, 12]
        cell_4_bias_parent = [7, 0, 12, 16]
        cell_5_bias_parent = [8, 10, 7, 3]
        cell_6_bias_parent = [8, 2, 5, 10]
        cell_7_bias_parent = [2, 6, 9, 5]
        cell_8_bias_parent = [9, 6, 1, 4]
        cell_9_bias_parent = [4, 1, 13, 15]
        cell_10_bias_parent = [12, 15, 13, 11]
        cell_11_bias_parent = [16, 12, 11, 14]
        cell_12_bias_parent = [3, 7, 16, 14]
        if distance_parent1_child11 < distance_parent1_child2:
            cell_1_bias_child = [9, 4, 0, 5]
            cell_2_bias_child = [4, 15, 12, 0]
            cell_3_bias_child = [0, 12, 16, 7]
            cell_4_bias_child = [5, 0, 7, 10]
            cell_5_bias_child = [6, 9, 5, 2]
            cell_6_bias_child = [6, 1, 4, 9]
            cell_7_bias_child = [1, 13, 15, 4]
            cell_8_bias_child = [15, 13, 11, 12]
            cell_9_bias_child = [12, 11, 14, 16]
            cell_10_bias_child = [7, 16, 14, 3]
            cell_11_bias_child = [10, 7, 3, 8]
            cell_12_bias_child = [2, 5, 10, 8]
        else:
            cell_1_bias_child = [15, 4, 0, 12]
            cell_2_bias_child = [4, 9, 5, 0]
            cell_3_bias_child = [0, 5, 10, 7]
            cell_4_bias_child = [12, 0, 7, 16]
            cell_5_bias_child = [13, 15, 12, 11]
            cell_6_bias_child = [13, 1, 4, 15]
            cell_7_bias_child = [1, 6, 9, 4]
            cell_8_bias_child = [9, 6, 2, 5]
            cell_9_bias_child = [5, 2, 8, 10]
            cell_10_bias_child = [7, 10, 8, 3]
            cell_11_bias_child = [16, 7, 3, 14]
            cell_12_bias_child = [11, 12, 16, 14]
        
    elif min_idx == 2:
        # parent 2 is closest to child 11
        parent_idx1_coord = coordinates[parent_idx + 1]
        distance_parent1_child3 = np.linalg.norm(parent_idx1_coord - child_idx3_coord)
        distance_parent1_child1 = np.linalg.norm(parent_idx1_coord - child_idx1_coord)
        cell_1_bias_parent = [10, 5, 0, 7]
        cell_2_bias_parent = [5,9,4,0]
        cell_3_bias_parent = [0, 4, 15, 12]
        cell_4_bias_parent = [7, 0, 12, 16]
        cell_5_bias_parent = [8, 10, 7, 3]
        cell_6_bias_parent = [8, 2, 5, 10]
        cell_7_bias_parent = [2, 6, 9, 5]
        cell_8_bias_parent = [9, 6, 1, 4]
        cell_9_bias_parent = [4, 1, 13, 15]
        cell_10_bias_parent = [12, 15, 13, 11]
        cell_11_bias_parent = [16, 12, 11, 14]
        cell_12_bias_parent = [3, 7, 16, 14]

        if distance_parent1_child3 < distance_parent1_child1:
            cell_1_bias_child = [15, 12, 0, 4]
            cell_2_bias_child = [12, 16, 7, 0]
            cell_3_bias_child = [0, 7, 10, 5]
            cell_4_bias_child = [4, 0, 5, 9]
            cell_5_bias_child = [13, 15, 4, 1]
            cell_6_bias_child = [13, 11, 12, 15]
            cell_7_bias_child = [11, 14, 16, 12]
            cell_8_bias_child = [16, 14, 3, 7]
            cell_9_bias_child = [7, 3, 8, 10]
            cell_10_bias_child = [5, 10, 8, 2]
            cell_11_bias_child = [9, 5, 2, 6]
            cell_12_bias_child = [1, 4, 9, 6]
        else:
            cell_1_bias_child = [16, 12, 0, 7]
            cell_2_bias_child = [12, 15, 4, 0]
            cell_3_bias_child = [0, 4, 9, 5]
            cell_4_bias_child = [7, 0, 5, 10]
            cell_5_bias_child = [14, 16, 7, 3]
            cell_6_bias_child = [14, 11, 12, 16]
            cell_7_bias_child = [11, 13, 15, 12]
            cell_8_bias_child = [15, 13, 1, 4]
            cell_9_bias_child = [4, 1, 6, 9]
            cell_10_bias_child = [5, 9, 6, 2]
            cell_11_bias_child = [10, 5, 2, 8]
            cell_12_bias_child = [3, 7, 10 ,8]
        
    elif min_idx == 3:
        # parent 2 is closest to child 3
        parent_idx1_coord = coordinates[parent_idx + 1]
        distance_parent1_child2 = np.linalg.norm(parent_idx1_coord - child_idx2_coord)
        distance_parent1_child11 = np.linalg.norm(parent_idx1_coord - child_idx11_coord)
        cell_1_bias_parent = [10, 5, 0, 7]
        cell_2_bias_parent = [5,9,4,0]
        cell_3_bias_parent = [0, 4, 15, 12]
        cell_4_bias_parent = [7, 0, 12, 16]
        cell_5_bias_parent = [8, 10, 7, 3]
        cell_6_bias_parent = [8, 2, 5, 10]
        cell_7_bias_parent = [2, 6, 9, 5]
        cell_8_bias_parent = [9, 6, 1, 4]
        cell_9_bias_parent = [4, 1, 13, 15]
        cell_10_bias_parent = [12, 15, 13, 11]
        cell_11_bias_parent = [16, 12, 11, 14]
        cell_12_bias_parent = [3, 7, 16, 14]

        if distance_parent1_child2 < distance_parent1_child11:
            cell_1_bias_child = [16, 7, 0, 12]
            cell_2_bias_child = [7, 10, 5, 0]
            cell_3_bias_child = [0, 5, 9, 4]
            cell_4_bias_child = [12, 0, 4, 15]
            cell_5_bias_child = [14, 16, 12, 11]
            cell_6_bias_child = [14, 3, 7, 16]
            cell_7_bias_child = [3, 8, 10, 7]
            cell_8_bias_child = [10, 8, 2, 5]
            cell_9_bias_child = [5, 2, 6, 9]
            cell_10_bias_child = [4, 9, 6, 1]
            cell_11_bias_child = [15, 4, 1, 13]
            cell_12_bias_child = [11, 12, 15, 13]
        else:
            cell_1_bias_child = [10, 7, 0, 5]
            cell_2_bias_child = [7, 16, 12, 0]
            cell_3_bias_child = [0, 12, 15, 4]
            cell_4_bias_child = [5, 0, 4, 9]
            cell_5_bias_child = [8, 10, 5, 2]
            cell_6_bias_child = [8, 3, 7, 10]
            cell_7_bias_child = [3, 14, 16, 7]
            cell_8_bias_child = [16, 14, 11, 12]
            cell_9_bias_child = [12, 11, 13, 15]
            cell_10_bias_child = [4, 15, 13, 1]
            cell_11_bias_child = [9, 4, 1, 6]
            cell_12_bias_child = [2, 5, 9, 6]
    else:
        raise ValueError("Cannot coonect p2p")

    def generate_cell_indices(bias_parent, bias_child):
        return [8] + [i + parent_idx for i in bias_parent] + [i + child_idx for i in bias_child]

    cell_indices_cell_1 = generate_cell_indices(cell_1_bias_parent, cell_1_bias_child)
    cell_indices_cell_2 = generate_cell_indices(cell_2_bias_parent, cell_2_bias_child)
    cell_indices_cell_3 = generate_cell_indices(cell_3_bias_parent, cell_3_bias_child)
    cell_indices_cell_4 = generate_cell_indices(cell_4_bias_parent, cell_4_bias_child)
    cell_indices_cell_5 = generate_cell_indices(cell_5_bias_parent, cell_5_bias_child)
    cell_indices_cell_6 = generate_cell_indices(cell_6_bias_parent, cell_6_bias_child)
    cell_indices_cell_7 = generate_cell_indices(cell_7_bias_parent, cell_7_bias_child)
    cell_indices_cell_8 = generate_cell_indices(cell_8_bias_parent, cell_8_bias_child)
    cell_indices_cell_9 = generate_cell_indices(cell_9_bias_parent, cell_9_bias_child)
    cell_indices_cell_10 = generate_cell_indices(cell_10_bias_parent, cell_10_bias_child)
    cell_indices_cell_11 = generate_cell_indices(cell_11_bias_parent, cell_11_bias_child)
    cell_indices_cell_12 = generate_cell_indices(cell_12_bias_parent, cell_12_bias_child)

    cell_indices = cell_indices_cell_1 + cell_indices_cell_2 + cell_indices_cell_3 + cell_indices_cell_4 + cell_indices_cell_5 + cell_indices_cell_6 + cell_indices_cell_7 + cell_indices_cell_8 + cell_indices_cell_9 + cell_indices_cell_10 + cell_indices_cell_11 + cell_indices_cell_12
    return np.array(cell_indices)

def create_cell_p2b(parent_idx, child_idx, coordinates):
    parent_idx = int(parent_idx)
    child_idx = int(child_idx)
    middle_points_bias_idx = [5, 12, 18]
    middle_points_idx = [i + child_idx for i in middle_points_bias_idx]
    parent_coordinate = coordinates[parent_idx]
    middle_point_distances = np.zeros(3)
    for i, middle_point_idx in enumerate(middle_points_idx):
        middle_point_coordinate = coordinates[middle_point_idx]
        distance = np.linalg.norm(middle_point_coordinate - parent_coordinate)
        middle_point_distances[i] = distance
    # get the minium two distances idx
    min_idx = np.argsort(middle_point_distances)[:2]
    # three cases
    # if min_idx is 0,1 or 1,0
    if (min_idx[0] == 0 and min_idx[1] == 1) or (min_idx[0] == 1 and min_idx[1] == 0):
        cell_1_bias = [10, 5, 0, 7]
        cell_2_bias = [5,9,4,0]
        cell_3_bias = [0, 4, 15, 12]
        cell_4_bias = [7, 0, 12, 16]
        cell_5_bias = [8, 10, 7, 3]
        cell_6_bias = [8, 2, 5, 10]
        cell_7_bias = [2, 6, 9, 5]
        cell_8_bias = [9, 6, 1, 4]
        cell_9_bias = [4, 1, 13, 15]
        cell_10_bias = [12, 15, 13, 11]
        cell_11_bias = [16, 12, 11, 14]
        cell_12_bias = [3, 7, 16, 14]

    elif (min_idx[0] == 1 and min_idx[1] == 2) or (min_idx[0] == 2 and min_idx[1] == 1):
        cell_1_bias = [16,12,0,7]
        cell_2_bias = [12,15,4,0]
        cell_3_bias = [0,4,21,18]
        cell_4_bias = [7,0,18,22]
        cell_5_bias = [14,15,7,3]
        cell_6_bias = [14,11,12,16]
        cell_7_bias = [11,13,15,12]
        cell_8_bias = [15,13,1,4]
        cell_9_bias = [4,1,19,21]
        cell_10_bias = [18,21,19,17]
        cell_11_bias = [22,18,17,20]
        cell_12_bias = [3,7,22,20]
    
    elif (min_idx[0] == 2 and min_idx[1] == 0) or (min_idx[0] == 0 and min_idx[1] == 2):
        cell_1_bias = [10,5,0,7]
        cell_2_bias = [5,9,4,0]
        cell_3_bias = [0,4,21,18]
        cell_4_bias = [7,0,18,22]
        cell_5_bias = [8,10,7,3]
        cell_6_bias = [8,2,5,10]
        cell_7_bias = [2,6,9,5]
        cell_8_bias = [9,6,1,4]
        cell_9_bias = [4,1,19,21]
        cell_10_bias = [18,21,19,17]
        cell_11_bias = [22,18,17,20]
        cell_12_bias = [3,7,22,20]
    
    else:
        raise ValueError("Invalid min_idx values")

    def generate_cell_indices(bias_parent, bias_child):
        return [8] + [i + parent_idx for i in bias_parent] + [i + child_idx for i in bias_child]

    cell_1_bias_parent = [10, 5, 0, 7]
    cell_2_bias_parent = [5,9,4,0]
    cell_3_bias_parent = [0, 4, 15, 12]
    cell_4_bias_parent = [7, 0, 12, 16]
    cell_5_bias_parent = [8, 10, 7, 3]
    cell_6_bias_parent = [8, 2, 5, 10]
    cell_7_bias_parent = [2, 6, 9, 5]
    cell_8_bias_parent = [9, 6, 1, 4]
    cell_9_bias_parent = [4, 1, 13, 15]
    cell_10_bias_parent = [12, 15, 13, 11]
    cell_11_bias_parent = [16, 12, 11, 14]
    cell_12_bias_parent = [3, 7, 16, 14]

    cell_indices_cell_1 = generate_cell_indices(cell_1_bias_parent, cell_1_bias)
    cell_indices_cell_2 = generate_cell_indices(cell_2_bias_parent, cell_2_bias)
    cell_indices_cell_3 = generate_cell_indices(cell_3_bias_parent, cell_3_bias)
    cell_indices_cell_4 = generate_cell_indices(cell_4_bias_parent, cell_4_bias)
    cell_indices_cell_5 = generate_cell_indices(cell_5_bias_parent, cell_5_bias)
    cell_indices_cell_6 = generate_cell_indices(cell_6_bias_parent, cell_6_bias)
    cell_indices_cell_7 = generate_cell_indices(cell_7_bias_parent, cell_7_bias)
    cell_indices_cell_8 = generate_cell_indices(cell_8_bias_parent, cell_8_bias)
    cell_indices_cell_9 = generate_cell_indices(cell_9_bias_parent, cell_9_bias)
    cell_indices_cell_10 = generate_cell_indices(cell_10_bias_parent, cell_10_bias)
    cell_indices_cell_11 = generate_cell_indices(cell_11_bias_parent, cell_11_bias)
    cell_indices_cell_12 = generate_cell_indices(cell_12_bias_parent, cell_12_bias)
    cell_indices = cell_indices_cell_1 + cell_indices_cell_2 + cell_indices_cell_3 + cell_indices_cell_4 + cell_indices_cell_5 + cell_indices_cell_6 + cell_indices_cell_7 + cell_indices_cell_8 + cell_indices_cell_9 + cell_indices_cell_10 + cell_indices_cell_11 + cell_indices_cell_12
    return np.array(cell_indices)

def create_cell_b2p(parent_idx, child_idx, coordinates):
    parent_idx = int(parent_idx)
    child_idx = int(child_idx)
    middle_points_bias_idx = [5, 12, 18]
    middle_points_idx = [i + parent_idx for i in middle_points_bias_idx]
    child_coordinate = coordinates[child_idx]
    middle_point_distances = np.zeros(3)
    for i, middle_point_idx in enumerate(middle_points_idx):
        middle_point_coordinate = coordinates[middle_point_idx]
        distance = np.linalg.norm(middle_point_coordinate - child_coordinate)
        middle_point_distances[i] = distance
    # get the minium two distances idx
    min_idx = np.argsort(middle_point_distances)[:2]
    # three cases
    # if min_idx is 0,1 or 1,0
    if (min_idx[0] == 0 and min_idx[1] == 1) or (min_idx[0] == 1 and min_idx[1] == 0):
        cell_1_bias = [10, 5, 0, 7]
        cell_2_bias = [5,9,4,0]
        cell_3_bias = [0, 4, 15, 12]
        cell_4_bias = [7, 0, 12, 16]
        cell_5_bias = [8, 10, 7, 3]
        cell_6_bias = [8, 2, 5, 10]
        cell_7_bias = [2, 6, 9, 5]
        cell_8_bias = [9, 6, 1, 4]
        cell_9_bias = [4, 1, 13, 15]
        cell_10_bias = [12, 15, 13, 11]
        cell_11_bias = [16, 12, 11, 14]
        cell_12_bias = [3, 7, 16, 14]

    elif (min_idx[0] == 1 and min_idx[1] == 2) or (min_idx[0] == 2 and min_idx[1] == 1):

        cell_4_bias = [16,12,0,7]
        cell_3_bias = [12,15,4,0]
        cell_2_bias = [0,4,21,18]
        cell_1_bias = [7,0,18,22]
        cell_12_bias = [14,16,7,3]
        cell_11_bias = [14,11,12,16]
        cell_10_bias = [11,13,15,12]
        cell_9_bias = [15,13,1,4]
        cell_8_bias = [4,1,19,21]
        cell_7_bias = [18,21,19,17]
        cell_6_bias = [22,18,17,20]
        cell_5_bias = [3,7,22,20]

        # reverse order of the list
        cell_1_bias = cell_1_bias[::-1]
        cell_2_bias = cell_2_bias[::-1]
        cell_3_bias = cell_3_bias[::-1]
        cell_4_bias = cell_4_bias[::-1]
        cell_5_bias = cell_5_bias[::-1]
        cell_6_bias = cell_6_bias[::-1]
        cell_7_bias = cell_7_bias[::-1]
        cell_8_bias = cell_8_bias[::-1]
        cell_9_bias = cell_9_bias[::-1]
        cell_10_bias = cell_10_bias[::-1]
        cell_11_bias = cell_11_bias[::-1]
        cell_12_bias = cell_12_bias[::-1]

    elif (min_idx[0] == 2 and min_idx[1] == 0) or (min_idx[0] == 0 and min_idx[1] == 2):
        cell_1_bias = [10,5,0,7]
        cell_2_bias = [5,9,4,0]
        cell_3_bias = [0,4,21,18]
        cell_4_bias = [7,0,18,22]
        cell_5_bias = [8,10,7,3]
        cell_6_bias = [8,2,5,10]
        cell_7_bias = [2,6,9,5]
        cell_8_bias = [9,6,1,4]
        cell_9_bias = [4,1,19,21]
        cell_10_bias = [18,21,19,17]
        cell_11_bias = [22,18,17,20]
        cell_12_bias = [3,7,22,20]
    
    else:
        raise ValueError("Invalid min_idx values")

    def generate_cell_indices(bias_parent, bias_child):
        return [8] + [i + parent_idx for i in bias_parent] + [i + child_idx for i in bias_child]

    cell_1_bias_child = [10, 5, 0, 7]
    cell_2_bias_child = [5,9,4,0]
    cell_3_bias_child = [0, 4, 15, 12]
    cell_4_bias_child = [7, 0, 12, 16]
    cell_5_bias_child = [8, 10, 7, 3]
    cell_6_bias_child = [8, 2, 5, 10]
    cell_7_bias_child = [2, 6, 9, 5]
    cell_8_bias_child = [9, 6, 1, 4]
    cell_9_bias_child = [4, 1, 13, 15]
    cell_10_bias_child = [12, 15, 13, 11]
    cell_11_bias_child = [16, 12, 11, 14]
    cell_12_bias_child = [3, 7, 16, 14]

    cell_indices_cell_1 = generate_cell_indices(cell_1_bias, cell_1_bias_child)
    cell_indices_cell_2 = generate_cell_indices(cell_2_bias, cell_2_bias_child)
    cell_indices_cell_3 = generate_cell_indices(cell_3_bias, cell_3_bias_child)
    cell_indices_cell_4 = generate_cell_indices(cell_4_bias, cell_4_bias_child)
    cell_indices_cell_5 = generate_cell_indices(cell_5_bias, cell_5_bias_child)
    cell_indices_cell_6 = generate_cell_indices(cell_6_bias, cell_6_bias_child)
    cell_indices_cell_7 = generate_cell_indices(cell_7_bias, cell_7_bias_child)
    cell_indices_cell_8 = generate_cell_indices(cell_8_bias, cell_8_bias_child)
    cell_indices_cell_9 = generate_cell_indices(cell_9_bias, cell_9_bias_child)
    cell_indices_cell_10 = generate_cell_indices(cell_10_bias, cell_10_bias_child)
    cell_indices_cell_11 = generate_cell_indices(cell_11_bias, cell_11_bias_child)
    cell_indices_cell_12 = generate_cell_indices(cell_12_bias, cell_12_bias_child)
    cell_indices = cell_indices_cell_1 + cell_indices_cell_2 + cell_indices_cell_3 + cell_indices_cell_4 + cell_indices_cell_5 + cell_indices_cell_6 + cell_indices_cell_7 + cell_indices_cell_8 + cell_indices_cell_9 + cell_indices_cell_10 + cell_indices_cell_11 + cell_indices_cell_12
    return np.array(cell_indices)