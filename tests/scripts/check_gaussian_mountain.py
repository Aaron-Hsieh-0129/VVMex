#!/usr/bin/env python3
"""Check an earlier Cartesian Gaussian-mountain BP5 history and its configuration."""
import argparse
import json

import adios2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("history", nargs="?", default="output/topo/vvm_output.bp")
parser.add_argument("--config", required=True, help="Cartesian gaussian_mountain configuration (not the RLL topography.json)")
args = parser.parse_args()
with open(args.config) as file:
    config = json.load(file)
grid = config["grid"]
mountain = config["initial_conditions"]["gaussian_mountain"]
x = (np.arange(grid["nx"])+.5)*grid["dx"]
length = grid["nx"]*grid["dx"]
distance = (x-mountain["center_x_m"]+length/2) % length-length/2
height = mountain["height_m"]*np.exp(-(distance/mountain["half_width_m"])**2)
expected_height = np.rint(height/grid["dz"])*grid["dz"]
expected_height = np.broadcast_to(expected_height, (grid["ny"], grid["nx"]))
with adios2.FileReader(args.history) as reader:
    variables = reader.available_variables()
    steps = int(variables["model_time_s"]["AvailableStepsCount"])
    times = []
    for step in range(steps):
        time = float(np.asarray(reader.read("model_time_s", step_selection=[step, 1])).squeeze())
        times.append(time)
        for name in config["output"]["fields_to_output"]:
            value = reader.read(name, step_selection=[step, 1])
            assert np.isfinite(value).all(), (step, name, "nonfinite")
        terrain = reader.read("terrain_height", step_selection=[step, 1])
        np.testing.assert_allclose(terrain, expected_height, atol=1e-12, rtol=0)
        topo = reader.read("topo", step_selection=[step, 1])
        np.testing.assert_array_equal(topo, expected_height/grid["dz"]+grid["n_halo_cells"]-1)
        mask = reader.read("ITYPEW", step_selection=[step, 1])
        levels = np.arange(grid["nz"])[:, None, None]+grid["n_halo_cells"]
        np.testing.assert_array_equal(mask, (levels > topo).astype(float))
    np.testing.assert_allclose(times, np.arange(0, config["simulation"]["total_time_s"]+1,
                                               config["simulation"]["output_interval_s"]), atol=0, rtol=0)
    w = reader.read("w", step_selection=[steps-1, 1])
    assert np.max(np.abs(w)) > 1e-6, "Terrain did not generate vertical motion"
    print(json.dumps({"passed": True, "saved_steps": steps, "final_time_s": times[-1],
                      "terrain_max_m": float(terrain.max()), "final_w_max_abs_m_s": float(np.max(np.abs(w)))}, indent=2))
