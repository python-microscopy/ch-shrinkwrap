import uuid

import yaml
from PYME.cluster.rules import RecipeRule

from evaluation_utils import testing_parameters

def generate_pointclouds(test_d, output_dir):
    test_pointcloud_id = uuid.uuid4()   # points are jittered by psf width
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
    """

    # pyme-cluster:///file_name

    rule = RecipeRule(recipe=recipe_text, output_dir='pyme-cluster:///'+output_dir, inputs={'__sim':['1']})

    rule.push()

    return test_pointcloud_id

def generate_test_shapes(test_d, output_dir):
    shape_pointcloud_id = uuid.uuid4()  # points are directly on the surface of the shape
    recipe_text = f"""
    - simulation.PointcloudFromShape:
        shape_name: {test_d['shape_name']}
        shape_params: {test_d['shape_params']}
        density: {test_d['density']}
        no_jitter: true
        output: shape
        p: 1.0
    - output.HDFOutput:
        filePattern: '{{output_dir}}/shape_{shape_pointcloud_id}.hdf'
        inputVariables:
            shape: shape
        scheme: pyme-cluster://
    """

    rule = RecipeRule(recipe=recipe_text, output_dir='pyme-cluster:///'+output_dir, inputs={'__sim':['1']})

    rule.push()

    return shape_pointcloud_id

def compute_shrinkwrap(test_d, output_dir, test_pointcloud_id, shape_pointcloud_id):
    shrinkwrap_stl_id = uuid.uuid4()
    recipe_text = f"""
    - pointcloud.Octree:
        input_localizations: test_test
        output_octree: octree
    - surface_fitting.DualMarchingCubes:
        input: octree
        output: mesh
        remesh: true
        threshold_density: {float(test_d['threshold_density']):.3e}
        n_points_min: {test_d['n_points_min']}
    - surface_fitting.ShrinkwrapMembrane:
        input: mesh
        max_iters: {test_d['max_iter']}
        curvature_weight: {test_d['curvature_weight']}      
        remesh_frequency: {test_d['remesh_frequency']}
        punch_frequency: {test_d['punch_frequency']}
        min_hole_radius: {test_d['min_hole_radius']}
        neck_threshold_low: {float(test_d['neck_threshold_low']):.3e}
        neck_threshold_high: {float(test_d['neck_threshold_high']):.3e}
        neck_first_iter: {test_d['neck_first_iter']}
        output: membrane
        points: test_test
    - surface_feature_extraction.PointsFromMesh:
        input: membrane
        output: membrane0_localizations
    - surface_feature_extraction.AverageSquaredDistance:
        input: membrane0_localizations
        input2: shape_shape
        output: average_squared_distance
    - simulation.AddAllMetadataToPipeline:
        inputMeasurements: average_squared_distance
        outputName: measurements
        additionalKeys: test_pointcloud_id shape_pointcloud_id shrinkwrap_stl_id
        additionalValues: {test_pointcloud_id} {shape_pointcloud_id} {shrinkwrap_stl_id}
    - output.HDFOutput:
        filePattern: '{{output_dir}}/sw_res.hdf'
        inputVariables:
            measurements: measurements
        scheme: pyme-cluster:// - aggregate
    - output.STLOutput:
        filePattern: '{{output_dir}}/sw_{shrinkwrap_stl_id}.stl'
        inputName: membrane
        scheme: pyme-cluster://
    """
    rule = RecipeRule(recipe=recipe_text, output_dir='pyme-cluster:///'+output_dir, 
                      inputs={'test': [f'pyme-cluster:///{output_dir}/test_{test_pointcloud_id}.hdf'],
                              'shape': [f'pyme-cluster:///{output_dir}/shape_{shape_pointcloud_id}.hdf']})

    rule.push()

    return shrinkwrap_stl_id

def compute_spr(test_d, output_dir, test_pointcloud_id, shape_pointcloud_id):
    spr_stl_id = uuid.uuid4()
    recipe_text = f"""
    - surface_fitting.ScreenedPoissonMesh:
        input: test_test
        depth: 12
        samplespernode: {test_d['samplespernode']}
        pointweight: {test_d['pointweight']}
        iters: {test_d['iters']}
        k: {test_d['k']}
        output: membrane
    - surface_feature_extraction.PointsFromMesh:
        input: membrane
        output: membrane0_localizations
    - surface_feature_extraction.AverageSquaredDistance:
        input: membrane0_localizations
        input2: shape_shape
        output: average_squared_distance
    - simulation.AddAllMetadataToPipeline:
        inputMeasurements: average_squared_distance
        outputName: measurements
        additionalKeys: test_pointcloud_id shape_pointcloud_id spr_stl_id
        additionalValues: {test_pointcloud_id} {shape_pointcloud_id} {spr_stl_id}
    - output.HDFOutput:
        filePattern: '{{output_dir}}/spr_res.hdf'
        inputVariables:
            measurements: measurements
        scheme: pyme-cluster:// - aggregate
    - output.STLOutput:
        filePattern: '{{output_dir}}/spr_{spr_stl_id}.stl'
        inputName: membrane
        scheme: pyme-cluster://
    """
    rule = RecipeRule(recipe=recipe_text, output_dir='pyme-cluster:///'+output_dir, 
                      inputs={'test': [f'pyme-cluster:///{output_dir}/test_{test_pointcloud_id}.hdf'],
                              'shape': [f'pyme-cluster:///{output_dir}/shape_{shape_pointcloud_id}.hdf']})

    rule.push()

    return spr_stl_id

def evaluate(file_name, generated_shapes_filename=None, technical_replicates=1):
    with open(file_name) as f:
        test_d = yaml.safe_load(f)

    sw_dicts, spr_dicts = testing_parameters(test_d)

    shape_dict = {}

    if generated_shapes_filename is None:
        ids = []
        for d in sw_dicts:
            for _ in range(technical_replicates):
                test_pointcloud_id = generate_pointclouds(d, test_d['save_fp'])

                # Only generate comparative shapes as necessary
                k = f"{d['shape_name']}_{'_'.join([f'{k}_{v}' for k,v in d['shape_params'].items()])}_{d['density']}"
                if k in shape_dict.keys():
                    shape_pointcloud_id = shape_dict[k]
                else:
                    shape_pointcloud_id = generate_test_shapes(d, test_d['save_fp'])
                    shape_dict[k] = shape_pointcloud_id
                
                ids.append({'test_id' : str(test_pointcloud_id), 'shape_id': str(shape_pointcloud_id)})
        with open('test_ids.yaml', 'w') as f:
            yaml.safe_dump([*ids], f)
    else:
        with open(generated_shapes_filename) as f:
            ids = yaml.safe_load(f)
        for id, d in zip(ids, sw_dicts):
            shrinkwrap_stl_id = compute_shrinkwrap(d, test_d['save_fp'], id['test_id'], id['shape_id'])
        for id, d in zip(ids, spr_dicts):
            spr_stl_id = compute_spr(d, test_d['save_fp'], id['test_id'], id['shape_id'])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load a YAML.')
    parser.add_argument('filename', help="YAML file containing paramater space for evaluation.", 
                        default=None, nargs='?')
    parser.add_argument('generated_shapes_filename', help="File containing list of generated shape IDs.", 
                        default=None, nargs='?')
    parser.add_argument('-n', '--technical_replicates', help="How many of each example should we run?",
                        default=1, type=int)

    args = parser.parse_args()

    evaluate(args.filename, args.generated_shapes_filename, args.technical_replicates)
