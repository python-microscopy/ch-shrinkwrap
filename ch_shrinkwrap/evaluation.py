from cgi import test
import os
import uuid

import yaml
from PYME.cluster.rules import RecipeRule

from evaluation_utils import testing_parameters

def generate_pointclouds(test_d, output_dir, inputs=None):
    test_pointcloud_id = uuid.uuid4()   # points are jittered by psf width
    shape_pointcloud_id = uuid.uuid4()  # points are directly on the surface of the shape
    recipe_text = f"""
    - simulation.PointcloudFromShape:
        output: test
        shape_name: {test_d['shape_name']}
        shape_params: {test_d['shape_params']}
        density: {test_d['density']}
        psf_width_x: {test_d['psf_width'][0]}
        psf_width_y: {test_d['psf_width'][1]}
        psf_width_z: {test_d['psf_width'][2]}
        mean_photon_count: {test_d['mean_photon_count']}
        bg_photon_count: {test_d['bg_photon_count']}
        noise_fraction: {test_d['noise_fraction']}
        p: {test_d['p']}
    - output.HDFOutput:
        filePattern: '{{output_dir}}/test_{test_pointcloud_id}.hdf'
        inputVariables:
            test: test
        scheme: pyme-cluster://
    - simulation.PointcloudFromShape:
        shape_name: {test_d['shape_name']}
        shape_params: {test_d['shape_params']}
        density: {test_d['density']}
        no_jitter: true
        output: shape
        p: {test_d['p']}
    - output.HDFOutput:
        filePattern: '{{output_dir}}/shape_{shape_pointcloud_id}.hdf'
        inputVariables:
            shape: shape
        scheme: pyme-cluster://
    """

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir, inputs={'__sim':['1']})

    rule.push()

    return test_pointcloud_id, shape_pointcloud_id

def compute_shrinkwrap(test_d, output_dir, inputs=None):
    recipe_text = f"""
    - pointcloud.Octree:
        input_localizations: two_toruses
        output_octree: octree
    - surface_fitting.DualMarchingCubes:
        input: octree
        output: mesh
        remesh: true
        threshold_density: {test_d['threshold_density']}
        n_points_min: {test_d['n_points_min']}
    - surface_fitting.ShrinkwrapMembrane:
        input: mesh
        max_iters: {test_d['max_iter']}
        curvature_weight: {test_d['curvature_weight']}      
        remesh_frequency: {test_d['remesh_frequency']}
        punch_frequency: {test_d['punch_frequency']}
        min_hole_radius: {test_d['min_hole_radius']}
        neck_theshold_low: {test_d['neck_theshold_low']}
        neck_threshold_high: {test_d['neck_threshold_high']}
        neck_first_iter: {test_d['neck_first_iter']}
        output: membrane
        points: two_toruses
    - surface_feature_extraction.PointsFromMesh:
        input: membrane
        output: membrane0_localizations
    - surface_feature_extraction.AverageSquaredDistance:
        input: membrane0_localizations
        input2: two_toruses_raw
        output: average_squared_distance
    """

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir, inputs=inputs)

    rule.push()

def evaluate(file_name):
    with open(file_name) as f:
        test_d = yaml.safe_load(f)

    try:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(file_name), test_d['save_fp']))
    except:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), test_d['save_fp']))

    sw_dicts, spr_dicts = testing_parameters(test_d)

    for d in sw_dicts:
        test_pointcloud_id, shape_pointcloud_id = generate_pointclouds(d, save_dir)
        # print(test_pointcloud_id, shape_pointcloud_id)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load a YAML.')
    parser.add_argument('filename', help="YAML file containing paramater space for evaluation.", 
                        default=None, nargs='?')
    # parser.add_argument('input_filename', help="HDF file to pass as input.", 
    #                     default=None, nargs='?')

    args = parser.parse_args()

    evaluate(args.filename)
