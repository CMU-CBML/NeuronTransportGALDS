# NeuronTransportGALDS
## Overview
This repository contains the official code for the paper **"GALDS: A Graph-Autoencoder-based Latent Dynamics Surrogate model to predict neurite material transport"**. It includes all necessary scripts for data pre-processing, model training, prediction, and post-processing to replicate our results.

## Workflow & Usage
The workflow is divided into four main stages: data generation, preprocessing, model training, and post-processing.

### 1. Data generation
To simulate the baseline neurite transport, we follow the methodology from [1] using the code available at the [NeuronTransportIGA repository](https://github.com/CMU-CBML/NeuronTransportIGA).

For experiments involving a **spatially asymmetric attachment rate**, we have modified the original simulation code. These modifications are provided in the `/IGA` directory. Please consult the updated `manual.txt` within that folder for detailed instructions on running these custom simulations.


### 2. Data preprocessing
After generating the simulation data (VTK files), the next step is to preprocess it for the model. This involves extracting nodal data from the neurite cross-sections (using 17 and 24 nodes, as described in the paper) and consolidating them into compact `.h5` files.

The necessary scripts are located in the `/preprocess` folder.

**To run preprocessing:**

```bash
python3 tree_extraction.py --case <case_name> --thread 20 --transport 1
```

**Key Parameters:**

`--case`: Name of the simulation case to process.\
`--thread`: Number of parallel threads to use for faster processing.\
`--ns`: (Optional) `1` to extract Navier-Stokes results.\
`--transport`: `1` to extract concentration results (required for GALDS).
`--k`: `1` to extract k values\
`--remove_old`: `1` to clean the target directory before generation.\
`--overwirte`: `1` to overwrite existing files.\

> **Note:** Proper structuring of the raw simulation files is crucial. Please refer to the `README.md` inside the `/preprocess` directory for the required file hierarchy.

### 3. Model training
The training scripts for all components of the GALDS architecture are located in the `/training` directory. This includes individual scripts for the:\

1. Autoencoder (Sec 4.1)
2. Latent Space Transformation Module (Sec 4.2)
3. Latent Space System Dynamics Module (Sec 4.3)

To train all modules sequentially, simply execute the main shell script.

To start training:
```bash
sh train_all.sh
```

### 4. Result Post-processing
The trained GALDS model outputs predictions in a point cloud format. To facilitate analysis and visualization, we project these points back onto the original neurite mesh using linear interpolation.

The scripts for this final step are available in the `/post_processing` directory.

To run post-processing:

```bash
sh post_process.sh
```

## Python Libraries & Dependencies

This project relies on several key Python libraries. The core dependencies are listed in `requirements.txt`.

[1] A. Li, X. Chai, G. Yang, Y. J. Zhang. An Isogeometric Analysis Computational Platform for Material Transport Simulation in Complex Neurite Networks. Molecular & Cellular Biomechanics, 16(2):123-140, 2019.