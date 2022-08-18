import os
import uuid

import yaml
from PYME.cluster.rules import RecipeRule

from evaluation_utils import testing_parameters

def generate_pointclouds(test_d, output_dir):
    test_pointcloud_id = uuid.uuid4()   # points are jittered by psf with
    shape_pointcloud_id = uuid.uuid4()  # points are directly on the surface of the shape
    recipe_text = f"""
    - localisations.AddPipelineDerivedVars:
    inputEvents: ''
    inputFitResults: FitResults
    outputLocalizations: Localizations
    - localisations.ProcessColour:
        input: Localizations
        output: colour_mapped
    - tablefilters.FilterTable:
        filters:
        error_x:
        - 0
        - 30
        inputName: colour_mapped
        outputName: filtered_localizations
    - generate.PointcloudFromShape:
        input: filtered_localizations
        output: {test_pointcloud_id}
        psf_width_x: {test_d['psf_width'][0]}
        psf_width_y: {test_d['psf_width'][1]}
        psf_width_z: {test_d['psf_width'][2]}
    - generate.PointcloudFromShape:
        density: {test_d['density']}
        input: filtered_localizations
        no_jitter: true
        output: {shape_pointcloud_id}
        p: {test_d['p']}
    """

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir)

    rule.push()

    return test_pointcloud_id, shape_pointcloud_id

def evaluate(file_name):
    try:
        save_dir = os.path.abspath(os.path.dirname(file_name))
    except:
        save_dir = os.path.abspath(os.path.dirname(__file__))

    with open(file_name) as f:
        test_d = yaml.safe_load(f)

    sw_dicts, spr_dicts = testing_parameters(test_d)

    for d in sw_dicts:
        test_pointcloud_id, shape_pointcloud_id = generate_pointclouds(d, save_dir)
        print(test_pointcloud_id, shape_pointcloud_id)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load a YAML.')
    parser.add_argument('filename', help="YAML file containing paramater space for evaluation.", 
                        default=None, nargs='?')

    args = parser.parse_args()

    evaluate(args.filename)
