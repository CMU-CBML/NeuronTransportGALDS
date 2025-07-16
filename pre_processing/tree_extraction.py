from extract_funcs import extract_data_vp, extract_data_n, extract_data_k
import os
from concurrent.futures import ProcessPoolExecutor
import argparse
import datetime

# Example usage
# python main_parallel_wholeTree.py --case "k_dome" --thread 8 --ns 0 --transport 0 --k 1 --remove_old 0 --overwrite 0

parser = argparse.ArgumentParser()

parser.add_argument("--case", type=str, default="NMO_66748")
parser.add_argument("--thread", type=int, default=10)
parser.add_argument("--ns", type=int, default=0)
parser.add_argument("--transport", type=int, default=0)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--remove_old", type=int, default=0)
parser.add_argument("--overwrite", type=int, default=0)

args = parser.parse_args()

# Make current directory the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
case_name = args.case

"""
- NMO_XXXX
    - output_segments_swc
        - pipe
            - pipe_1.swc
            - ...
        - bifurcation
            - bifurcation_1.swc
            - ...
    - output_segments_vtu
        - pipe
        - bifurcation
    - sim_result
        - sim_0
            - parameter.txt (v_in, n0, ...)
            - tr_rsult
                - controlmesh_allparticle_0.vtk
                - ...
            -ns_result
                - controlmesh_VelocityPressure.vtk
        - sim_1
        - ...
    - Data
        - sim_0
            - parameter.txt (v_in, n0, ...)
            - pipe
                - sim_0_skeleton_0.vtk
                - ...
            - bifurcation
                - sim_0_skeleton_0.vtk
                - ...
            -bifurcation_ns
                - ns_skeleton_0.vtk
                - ...
            -pipe_ns
                - ns_skeleton_0.vtk
                - ...
            -wholeTree
                - wholeTree_ns.h5 (data for navier-stokes)
                - wholeTree_tr_step_0.h5 (data for transport)
                - wholeTree_tr_step_1.h5
                - ...
        - sim_1
        - ...
    - controlmesh.vtk
    -skeleton_smooth.swc
    -skeleton_initial.swc
"""

def process_skeleton(sim_step, sim_result_path, is_ns=False, sim_num=0):
    skeleton_path = f"{case_name}/skeleton_smooth.swc"
    if is_ns:
        sim_name = os.path.basename(sim_result_path)
        print(f"Processing NS files from ({sim_name}): {case_name}", flush=True)
        save_file_name = f"{case_name}/Data/sim_{sim_num}/wholeTree/wholeTree_ns"
        # Check if the file exists
        if (not os.path.exists(f"{save_file_name}.h5")) or args.overwrite:
            try:
                extract_data_vp(sim_result_path, skeleton_path, save_file_name)
            except Exception as e:
                print(e, flush=True)
                print(f"Cannot extract sim:{sim_name}", flush=True)
        else:
            print(f"File already exists: {sim_result_path} and overwite is set to False", flush=True)
    else:
        sim_name = os.path.basename(sim_result_path)
        print(f"Processing Transport files from ({sim_name}): {case_name}", flush=True)
        save_file_name = f"{case_name}/Data/sim_{sim_num}/wholeTree/wholeTree_tr_step_{sim_step}"
        # Check if the file exists
        if (not os.path.exists(f"{save_file_name}.h5")) or args.overwrite:
            try:
                extract_data_n(sim_result_path, skeleton_path, save_file_name)
            except Exception as e:
                print(e, flush=True)
                print(f"Cannot extract sim_name: {sim_name}", flush=True)  
        else:
            print(f"File already exists: {sim_result_path} and overwite is set to False", flush=True)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)    

def process_skeleton_k(sim_result_path, sim_num):
    skeleton_path = f"{case_name}/skeleton_smooth.swc"
    save_file_name = f"{case_name}/Data/k_values_{sim_num}"
    print("save file name:", save_file_name, flush=True)
    # Check if the file exists
    if (not os.path.exists(f"{save_file_name}.h5")) or args.overwrite:
        try:
            extract_data_k(sim_result_path, skeleton_path, save_file_name)
        except Exception as e:
            print(e, flush=True)
            print(f"Cannot extract sim:{sim_result_path}", flush=True)
    else:
        print(f"File already exists: {sim_result_path} and overwite is set to False", flush=True)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)    


def main():
    # Check number of simulations
    
    if args.ns or args.transport:
        sim_dir_base = f"{case_name}/sim_result"
        sub_dir_names = os.listdir(sim_dir_base)
        sim_num_ls = []
        for sub_dir_name in sub_dir_names:
            if sub_dir_name.startswith("sim"):
                sim_num_ls.append(sub_dir_name.split("_")[-1])
    elif args.k:
        sim_dir_base = f"{case_name}"
        sub_dir_names = os.listdir(sim_dir_base)
        sim_num_ls = []
        for sim_name in os.listdir(f"{case_name}"):
            if sim_name.startswith("k_values_"):
                sim_num_ls.append(sim_name.split("_")[-1].split(".")[0])
    print(f"This process will extract data from {case_name} with {len(sim_num_ls)} simulations")

    # Ensure directory structure exists
    if not os.path.exists(f"{case_name}/Data"):
        os.makedirs(f"{case_name}/Data")
    
    # Parallel data extraction
    with ProcessPoolExecutor(max_workers=args.thread) as executor:
        sim_name_ls = []
        tasks = []
        if args.ns or args.transport:
            for sim_num in sim_num_ls:
                if not os.path.exists(f"{case_name}/Data/sim_{sim_num}/wholeTree"):
                    os.makedirs(f"{case_name}/Data/sim_{sim_num}/wholeTree")
                if args.remove_old:
                    for file in os.listdir(f"{case_name}/Data/sim_{sim_num}/wholeTree"):
                        os.remove(f"{case_name}/Data/sim_{sim_num}/wholeTree/{file}")
                if args.ns:
                    sim_result_path = f"{case_name}/sim_result/sim_{sim_num}/ns_result/controlmesh_VelocityPressure.vtk"
                    tasks.append(executor.submit(process_skeleton, 0, sim_result_path, is_ns=True, sim_num=sim_num))
                if args.transport:
                    for sim_name in os.listdir(f"{case_name}/sim_result/sim_{sim_num}/tr_result"):
                        if sim_name.startswith("controlmesh_allparticle_"):
                            sim_name_ls.append(sim_name)
                    for _, sim_result_name in enumerate(sim_name_ls):
                        sim_step = int(sim_result_name.split("_")[-1].split(".")[0])
                        sim_result_path = f"{case_name}/sim_result/sim_{sim_num}/tr_result/{sim_result_name}"
                        tasks.append(executor.submit(process_skeleton, sim_step, sim_result_path, is_ns=False, sim_num=sim_num))
        elif args.k:
            for sim_name in os.listdir(f"{case_name}"):
                if sim_name.startswith("k_values_"):
                    sim_name_ls.append(sim_name)
            for _, sim_result_name in enumerate(sim_name_ls):
                sim_result_path = f"{case_name}/{sim_result_name}"
                sim_num = int(sim_result_name.split("_")[-1].split(".")[0])
                tasks.append(executor.submit(process_skeleton_k, sim_result_path, sim_num=sim_num))
            
        for task in tasks:
            task.result()

if __name__ == "__main__":
    main()
