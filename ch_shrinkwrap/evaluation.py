from cgi import test
import os
import uuid

import yaml
from PYME.cluster.rules import RecipeRule

from evaluation_utils import testing_parameters

def generate_pointclouds(test_d, output_dir):
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

    # pyme-cluster:///file_name

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir, inputs={'__sim':['1']})

    rule.push()

    return test_pointcloud_id, shape_pointcloud_id

def compute_shrinkwrap(test_d, output_dir, test_pointcloud_id, shape_pointcloud_id):
    shrinkwrap_pointcloud_id = uuid.uuid4()
    recipe_text = f"""
    - pointcloud.Octree:
        input_localizations: shape
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
        neck_threshold_low: {test_d['neck_threshold_low']}
        neck_threshold_high: {test_d['neck_threshold_high']}
        neck_first_iter: {test_d['neck_first_iter']}
        output: membrane
        points: two_toruses
    - surface_feature_extraction.PointsFromMesh:
        input: membrane
        output: membrane0_localizations
    - surface_feature_extraction.AverageSquaredDistance:
        input: membrane0_localizations
        input2: test
        output: average_squared_distance
    - output.HDFOutput:
        filePattern: '{{output_dir}}/sw_{shrinkwrap_pointcloud_id}.hdf'
        inputVariables:
            average_squared_distance: average_squared_distance
        scheme: pyme-cluster://
    - output.STLOutput:
        filePattern: '{{output_dir}}/sw_{shrinkwrap_pointcloud_id}.stl'
        inputVariables:
            membrane: membrane
        scheme: pyme-cluster://
    """

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir, inputs={'test': f'pyme-cluster://{output_dir}/test_{test_pointcloud_id}.hdf',
                                                                         'shape': f'pyme-cluster://{output_dir}/shape_{shape_pointcloud_id}.hdf'})

    rule.push()

    return shrinkwrap_pointcloud_id

def evaluate(file_name, generated_shapes_filename=None):
    with open(file_name) as f:
        test_d = yaml.safe_load(f)

    sw_dicts, spr_dicts = testing_parameters(test_d)

    if generated_shapes_filename is None:
        ids = []
        for d in sw_dicts:
            test_pointcloud_id, shape_pointcloud_id = generate_pointclouds(d, test_d['save_fp'])
            ids.append({'test_id' : str(test_pointcloud_id), 'shape_id': str(shape_pointcloud_id)})
        print(ids)
        with open('test_ids.yaml', 'w') as f:
            yaml.safe_dump([*ids], f)
    else:
        with open(generated_shapes_filename) as f:
            ids = yaml.safe_load(f)
        for id, d in zip(ids, sw_dicts):
            shrinkwrap_pointcloud_id = compute_shrinkwrap(d, test_d['save_fp'], id['test_id'], id['shape_id'])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load a YAML.')
    parser.add_argument('filename', help="YAML file containing paramater space for evaluation.", 
                        default=None, nargs='?')
    parser.add_argument('generated_shapes_filename', help="File containing list of generated shape IDs.", 
                        default=None, nargs='?')

    args = parser.parse_args()

    evaluate(args.filename, args.generated_shapes_filename)
