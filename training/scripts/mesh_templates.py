import numpy as np

index_mapping_circle_surface = {
    0 : np.array([0.000000, 0.000000, 0.000000]), # Project to index 0 in original mesh
    1 : np.array([-1.000000, 0.000000, 0.000000]), # Project to index 1 in original mesh
    2 : np.array([0.000000, 1.000000, 0.000000]), # Project to index 2 in original mesh
    3 : np.array([1.000000, 0.000000, 0.000000]), # Project to index 3 in original mesh
    4 : np.array([-0.571429, 0.000000, 0.000000]), # Project to index 7 in original mesh
    5 : np.array([0.000000, 0.571429, 0.000000]), # Project to index 13 in original mesh
    6 : np.array([-0.707107, 0.707107, 0.000000]), # Project to index 20 in original mesh
    7 : np.array([0.571429, 0.000000, 0.000000]), # Project to index 27 in original mesh
    8 : np.array([0.707107, 0.707107, 0.000000]), # Project to index 35 in original mesh
    9 : np.array([-0.517460, 0.519049, 0.000000]), # Project to index 62 in original mesh
    10 : np.array([0.519049, 0.517460, 0.000000]), # Project to index 96 in original mesh
    11 : np.array([0.000000, -1.000000, 0.000000]), # Project to index 108 in original mesh
    12 : np.array([0.000000, -0.571429, 0.000000]), # Project to index 112 in original mesh
    13 : np.array([-0.707107, -0.707107, 0.000000]), # Project to index 119 in original mesh
    14 : np.array([0.707107, -0.707107, 0.000000]), # Project to index 128 in original mesh
    15 : np.array([-0.517460, -0.519049, 0.000000]), # Project to index 155 in original mesh
    16 : np.array([0.519049, -0.517460, 0.000000]) # Project to index 189 in original mesh
}

index_extract_order_circle_surface = [i for i in range(17)]

index_mapping_bifurcation_surface = {
    0 : np.array([0.000000, 0.000000, 0.000000]), # Project to index 0 in original mesh
    1 : np.array([-1.000000, 0.000000, 0.000000]), # Project to index 1 in original mesh
    2 : np.array([0.000000, 0.866025, -0.500000]), # Project to index 2 in original mesh
    3 : np.array([1.000000, 0.000000, 0.000000]), # Project to index 3 in original mesh
    4 : np.array([-0.571429, 0.000000, 0.000000]), # Project to index 7 in original mesh
    5 : np.array([0.000000, 0.494872, -0.285714]), # Project to index 13 in original mesh
    6 : np.array([-0.707107, 0.612373, -0.353553]), # Project to index 20 in original mesh
    7 : np.array([0.571429, 0.000000, 0.000000]), # Project to index 27 in original mesh
    8 : np.array([0.707107, 0.612373, -0.353553]), # Project to index 35 in original mesh
    9 : np.array([-0.517460, 0.449510, -0.259524]), # Project to index 62 in original mesh
    10 : np.array([0.519049, 0.448134, -0.258730]), # Project to index 96 in original mesh
    11 : np.array([0.000000, -0.866025, -0.500000]), # Project to index 108 in original mesh
    12 : np.array([0.000000, -0.494872, -0.285714]), # Project to index 112 in original mesh
    13 : np.array([-0.707107, -0.612373, -0.353553]), # Project to index 119 in original mesh
    14 : np.array([0.707107, -0.612373, -0.353553]), # Project to index 128 in original mesh
    15 : np.array([-0.517460, -0.449510, -0.259524]), # Project to index 155 in original mesh
    16 : np.array([0.519049, -0.448134, -0.258730]), # Project to index 189 in original mesh
    17: np.array([0.000000, 0.000000, 1.000000]), # Project to index 201 in original mesh
    18: np.array([0.000000, 0.000000, 0.571429]), # Project to index 205 in original mesh
    19: np.array([-0.707107, 0.000000, 0.707107]), # Project to index 212 in original mesh
    20: np.array([0.707107, 0.000000, 0.707107]), # Project to index 221 in original mesh
    21: np.array([-0.517460, 0.000000, 0.519049]), # Project to index 248 in original mesh
    22: np.array([0.519049, 0.000000, 0.517460]), # Project to index 282 in original mesh
}

index_extract_order_bifurcation_surface = [i for i in range(23)]
