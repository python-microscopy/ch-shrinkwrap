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
        psf_width_x: {test_d['psf_width'][0]}
        psf_width_y: {test_d['psf_width'][1]}
        psf_width_z: {test_d['psf_width'][2]}
    - output.HDFOutput:
        filePattern: '{{output_dir}}/{test_pointcloud_id}.hdf'
        inputVariables:
            test: test
        scheme: pyme-cluster://
    - simulation.PointcloudFromShape:
        density: {test_d['density']}
        no_jitter: true
        output: shape
        p: {test_d['p']}
    - output.HDFOutput:
        filePattern: '{{output_dir}}/{shape_pointcloud_id}.hdf'
        inputVariables:
            shape: shape
        scheme: pyme-cluster://
    """

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir, inputs={'__sim':['1']})

    rule.push()

    return test_pointcloud_id, shape_pointcloud_id

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
