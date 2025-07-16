This readme is additional infromation directory format.
The simulation result is all under `MMO_XXXX\sim_result` and output .h5 file will be generated under `NMO_XXXX\Data`.

## Directory Structure
```
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
            - ns_result
                - controlmesh_VelocityPressure.vtk
        - sim_1
        - ...
    - Data
        - sim_0
            - parameter.txt (v_in, n0, ...)
            - wholeTree
                - wholeTree_ns.h5 (data for navier-stokes)
                - wholeTree_tr_step_0.h5 (data for transport)
                - wholeTree_tr_step_1.h5
                - ...
        - sim_1
        - ...
    - controlmesh.vtk
    - skeleton_smooth.swc
    - skeleton_initial.swc
```