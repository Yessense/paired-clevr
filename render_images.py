# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function

import math, sys, random, argparse, json, os, tempfile
from copy import copy
from datetime import datetime as dt
from collections import Counter

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector

    print("opened")
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='dataset/base_scene.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='dataset/properties.json',
                    help="JSON file defining objects, materials, sizes, and colors. " +
                         "The \"colors\" field maps from CLEVR color names to RGB values; " +
                         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                         "rescale object models; the \"materials\" and \"shapes\" fields map " +
                         "from CLEVR material and shape names to .blend files in the " +
                         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='dataset/shapes',
                    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='dataset/materials',
                    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default='/home/yessense/projects/paired-clevr/dataset/CoGenT_A.json',
                    help="Optional path to a JSON file mapping shape names to a list of " +
                         "allowed color names for that shape. This allows rendering images " +
                         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
                    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
                    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
                    help="Along all cardinal directions (left, right, front, back), all " +
                         "objects will be at least this distance apart. This makes resolving " +
                         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
                    help="All objects will have at least this many visible pixels in the " +
                         "final rendered images; this ensures that no objects are fully " +
                         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
                    help="The number of times to try placing an object before giving up and " +
                         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
                    help="The index at which to start for numbering rendered images. Setting " +
                         "this to non-zero values allows you to distribute rendering across " +
                         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=1, type=int,
                    help="The number of images to render")
parser.add_argument('--filename_prefix', default='paired',
                    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='clevr',
                    help="Name of the split for which we are rendering. This will be added to " +
                         "the names of rendered images, and will also be stored in the JSON " +
                         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../dataset/images/',
                    help="The directory where dataset images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../dataset/scenes/',
                    help="The directory where dataset JSON scene structures will be stored. " +
                         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../dataset/CLEVR_scenes.json',
                    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='dataset/blendfiles',
                    help="The directory where blender scene files will be stored, if the " +
                         "user requested that these files be saved using the " +
                         "--save_blendfiles flag; in this case it will be created if it does " +
                         "not already exist.")
parser.add_argument('--version', default='1.0',
                    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
                    default="Creative Commons Attribution (CC-BY 4.0)",
                    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                    help="String to store in the \"date\" field of the generated JSON file; " +
                         "defaults to today's date")

# ----------------------------------------
# Rendering options
# ----------------------------------------

# gpu
parser.add_argument('--use_gpu', default=1, type=int,
                    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")

# image size 320 * 240
parser.add_argument('--width', default=320, type=int,
                    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
                    help="The height (in pixels) for the rendered images")

# Random camera position
parser.add_argument('--key_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
                    help="The magnitude of random jitter to add to the camera position")

parser.add_argument('--render_num_samples', default=512, type=int,
                    help="The number of samples to use when rendering. Larger values will " +
                         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
                    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
                    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
                    help="The tile size to use for rendering. This should not affect the " +
                         "quality of the rendered image but may affect the speed; CPU-based " +
                         "rendering may achieve better performance using smaller tile sizes " +
                         "while larger tile sizes may be optimal for GPU-based rendering.")


def main(args):
    pair_creator = PairCreator(args)

    num_digits = 6

    img_extension = ".png"
    scene_extension = ".json"

    # create directories if not exist
    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)

    all_scene_paths = []
    for i in range(args.num_images):
        img_features, pair_features, index = pair_creator.create_pair_features()
        scene_path = render_image(img_features, pair_features,
                                  i + args.start_idx, pair_creator,
                                  args)
        all_scene_paths.append(scene_path)

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)


class PairCreator:
    def __init__(self, args):
        self.shape_color_combos = None

        if args.shape_color_combos_json is not None:
            with open(args.shape_color_combos_json, 'r') as f:
                self.shape_color_combos = json.load(f)

        with open(args.properties_json, 'r') as f:
            # Properties
            properties = json.load(f)

            # Color
            color_name_to_rgba = {}
            for name, rgb in properties['colors'].items():
                rgba = [float(c) / 255.0 for c in rgb] + [1.0]
                color_name_to_rgba[name] = rgba
            self.color_name_to_rgba_dict = color_name_to_rgba
            self.color_name_to_rgba = list(color_name_to_rgba.items())

            # Materials
            self.material_mapping = [(v, k) for k, v in properties['materials'].items()]
            # Shapes
            self.object_mapping = [(v, k) for k, v in properties['shapes'].items()]
            # Sizes
            self.size_mapping = list(properties['sizes'].items())

            # x, y spans
            self.x_span = (-3, 3)
            self.y_span = (-3, 3)

            # Orientation
            self.orientation = 0.0

            self.features_to_exchange = ['color', 'material', 'shape', 'size', 'x', 'y']

    def create_random_features(self):
        shape = random.choice(self.object_mapping)
        if self.shape_color_combos is None:
            color = random.choice(self.color_name_to_rgba)
        else:
            color_name = random.choice(self.shape_color_combos[shape[1]])
            color_rgba = self.color_name_to_rgba_dict[color_name]
            color = (color_name, color_rgba)

        img_features = Features(color=color,
                                material=random.choice(self.material_mapping),
                                shape=shape,
                                size=random.choice(self.size_mapping),
                                x=random.uniform(*self.x_span),
                                y=random.uniform(*self.y_span),
                                orientation=self.orientation)
        return img_features

    def create_pair_features(self):
        img_features = self.create_random_features()

        exchanged_feature = random.choice(self.features_to_exchange)

        paired_features = Features(color=copy(img_features.color),
                                   material=copy(img_features.material),
                                   shape=copy(img_features.shape),
                                   size=copy(img_features.size),
                                   x=img_features.x,
                                   y=img_features.y,
                                   orientation=img_features.orientation)
        if exchanged_feature == 'color':
            while paired_features.color == img_features.color:
                paired_features.color = random.choice(self.color_name_to_rgba)
        elif exchanged_feature == 'material':
            while paired_features.material == img_features.material:
                paired_features.material = random.choice(self.material_mapping)
        elif exchanged_feature == 'shape':
            while paired_features.shape == img_features.shape:
                paired_features.shape = random.choice(self.object_mapping)
        elif exchanged_feature == 'size':
            while paired_features.size == img_features.size:
                paired_features.size = random.choice(self.size_mapping)
        elif exchanged_feature == 'x':
            paired_features.x = random.uniform(-3, 3)
        elif exchanged_feature == 'y':
            paired_features.y = random.uniform(-3, 3)
        return img_features, paired_features, self.features_to_exchange.index(exchanged_feature)


class Features:
    def __init__(self, color, material, shape, size, x, y, orientation):
        self.orientation = orientation
        self.y = y
        self.x = x
        self.size = size
        self.shape = shape
        self.material = material
        self.color = color

    def __str__(self):
        return " ".join([str(value) for value in
                         [self.orientation, self.y, self.x, self.size, self.shape, self.material, self.color]])


def render_image(img_features,
                 pair_features,
                 output_index,
                 pair_creator,
                 args):
    output_scene = os.path.join(args.output_scene_dir, '%s_%06d.json' % ('scene', output_index))
    features = {'img': img_features, 'pair': pair_features}

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)
        # Add random jitter to camera position

    camera_jitter = [rand(args.camera_jitter) for i in range(3)]
    key_light_jitter = [rand(args.key_light_jitter) for i in range(3)]
    back_light_jitter = [rand(args.back_light_jitter) for i in range(3)]
    fill_light_jitter = [rand(args.fill_light_jitter) for i in range(3)]
    scenes_struct = []

    n_objects_on_scene = random.randint(args.min_objects - 2, args.max_objects - 2)

    for new_object_num in range(1, n_objects_on_scene + 1):
        add_image = True

        num_tries = 0
        while True:
            new_object_features = pair_creator.create_random_features()
            if num_tries > args.max_retries:
                add_image = False
                break
            else:
                dists_good = True
                for image_features in features.values():
                    xx, yy, rr = image_features.x, image_features.y, image_features.size[1]
                    x, y, r = new_object_features.x, new_object_features.y, new_object_features.size[1]
                    dx, dy = x - xx, y - yy
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist - r - rr < args.min_dist:
                        dists_good = False
                        break
                if dists_good:
                    break
                else:
                    num_tries += 1
                    continue
        if add_image:
            obj_name = 'obj{number}'.format(number=new_object_num)
            features[obj_name] = new_object_features

    for img in features:
        output_image = os.path.join(args.output_image_dir, '%s_%06d.png' % (img, output_index))
        # Load the main blendfile
        bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

        # Load materials
        utils.load_materials(args.material_dir)

        # Set render arguments so we can get pixel coordinates later.
        # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
        # cannot be used.
        render_args = bpy.context.scene.render
        render_args.engine = "CYCLES"
        render_args.filepath = output_image
        render_args.resolution_x = args.width
        render_args.resolution_y = args.height
        render_args.resolution_percentage = 100
        render_args.tile_x = args.render_tile_size
        render_args.tile_y = args.render_tile_size

        if args.use_gpu == 1:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

        # Some CYCLES-specific stuff
        bpy.data.worlds['World'].cycles.sample_as_light = True
        bpy.context.scene.cycles.blur_glossy = 2.0
        bpy.context.scene.cycles.samples = args.render_num_samples
        bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
        bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

        if args.use_gpu == 1:
            bpy.context.scene.cycles.device = 'GPU'

        # This will give ground-truth information about the scene and its objects
        scene_struct = {
            'image_index': output_index,
            'image_filename': os.path.basename(output_image),
            'objects': [],
            'directions': {},
        }

        # Put a plane on the ground so we can compute cardinal directions
        bpy.ops.mesh.primitive_plane_add(radius=5)
        plane = bpy.context.object

        # Add random jitter to camera position
        if args.camera_jitter > 0:
            for i, jitter in enumerate(camera_jitter):
                bpy.data.objects['Camera'].location[i] += jitter

        # Figure out the left, up, and behind directions along the plane and record
        # them in the scene structure
        camera = bpy.data.objects['Camera']
        plane_normal = plane.data.vertices[0].normal
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
        plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
        plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
        plane_up = cam_up.project(plane_normal).normalized()

        # Delete the plane; we only used it for normals anyway. The base scene file
        # contains the actual ground plane.
        utils.delete_object(plane)

        # Save all six axis-aligned directions in the scene struct
        scene_struct['directions']['behind'] = tuple(plane_behind)
        scene_struct['directions']['front'] = tuple(-plane_behind)
        scene_struct['directions']['left'] = tuple(plane_left)
        scene_struct['directions']['right'] = tuple(-plane_left)
        scene_struct['directions']['above'] = tuple(plane_up)
        scene_struct['directions']['below'] = tuple(-plane_up)

        # Add random jitter to lamp positions
        if args.key_light_jitter > 0:
            for i, jitter in enumerate(key_light_jitter):
                bpy.data.objects['Lamp_Key'].location[i] += jitter
        if args.back_light_jitter > 0:
            for i, jitter in enumerate(back_light_jitter):
                bpy.data.objects['Lamp_Back'].location[i] += jitter
        if args.fill_light_jitter > 0:
            for i, jitter in enumerate(fill_light_jitter):
                bpy.data.objects['Lamp_Fill'].location[i] += jitter

        # Now make some random objects
        object, blender_object = add_object(features[img], args, camera)

        # Render the scene and dump the scene dataset structure
        scene_struct['objects'] = object
        # scene_struct['relationships'] = compute_all_relationships(scene_struct)
        while True:
            try:
                bpy.ops.render.render(write_still=True)
                break
            except Exception as e:
                print(e)
        scenes_struct.append(scene_struct)

    features_img = {k: v for k, v in features.items() if k != 'pair'}
    features_pair = {k: v for k, v in features.items() if k != 'img'}
    scene_features = {'img': features_img, 'pair': features_pair}

    for scene_name, scene_features in scene_features.items():
        output_image = os.path.join(args.output_image_dir, '%s_%s_%06d.png' % ('scene', scene_name, output_index))
        # Load the main blendfile
        bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
        # Load materials
        utils.load_materials(args.material_dir)
        # Set render arguments so we can get pixel coordinates later.
        # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
        # cannot be used.
        render_args = bpy.context.scene.render
        render_args.engine = "CYCLES"
        render_args.filepath = output_image
        render_args.resolution_x = args.width
        render_args.resolution_y = args.height
        render_args.resolution_percentage = 100
        render_args.tile_x = args.render_tile_size
        render_args.tile_y = args.render_tile_size

        if args.use_gpu == 1:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

        # Some CYCLES-specific stuff
        bpy.data.worlds['World'].cycles.sample_as_light = True
        bpy.context.scene.cycles.blur_glossy = 2.0
        bpy.context.scene.cycles.samples = args.render_num_samples
        bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
        bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

        if args.use_gpu == 1:
            bpy.context.scene.cycles.device = 'GPU'

        # This will give ground-truth information about the scene and its objects
        scene_struct = {
            'image_index': output_index,
            'image_filename': os.path.basename(output_image),
            'objects': [],
            'directions': {},
        }

        # Put a plane on the ground so we can compute cardinal directions
        bpy.ops.mesh.primitive_plane_add(radius=5)
        plane = bpy.context.object

        # Add random jitter to camera position
        if args.camera_jitter > 0:
            for i, jitter in enumerate(camera_jitter):
                bpy.data.objects['Camera'].location[i] += jitter

        # Figure out the left, up, and behind directions along the plane and record
        # them in the scene structure
        camera = bpy.data.objects['Camera']
        plane_normal = plane.data.vertices[0].normal
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
        plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
        plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
        plane_up = cam_up.project(plane_normal).normalized()

        # Delete the plane; we only used it for normals anyway. The base scene file
        # contains the actual ground plane.
        utils.delete_object(plane)

        # Save all six axis-aligned directions in the scene struct
        scene_struct['directions']['behind'] = tuple(plane_behind)
        scene_struct['directions']['front'] = tuple(-plane_behind)
        scene_struct['directions']['left'] = tuple(plane_left)
        scene_struct['directions']['right'] = tuple(-plane_left)
        scene_struct['directions']['above'] = tuple(plane_up)
        scene_struct['directions']['below'] = tuple(-plane_up)

        # Add random jitter to lamp positions
        if args.key_light_jitter > 0:
            for i, jitter in enumerate(key_light_jitter):
                bpy.data.objects['Lamp_Key'].location[i] += jitter
        if args.back_light_jitter > 0:
            for i, jitter in enumerate(back_light_jitter):
                bpy.data.objects['Lamp_Back'].location[i] += jitter
        if args.fill_light_jitter > 0:
            for i, jitter in enumerate(fill_light_jitter):
                bpy.data.objects['Lamp_Fill'].location[i] += jitter

        # Now make some random objects
        object, blender_object = add_objects(scene_features, args, camera)

        # Render the scene and dump the scene dataset structure
        scene_struct['objects'] = object
        # scene_struct['relationships'] = compute_all_relationships(scene_struct)
        while True:
            try:
                bpy.ops.render.render(write_still=True)
                break
            except Exception as e:
                print(e)
        scenes_struct.append(scene_struct)

    with open(output_scene, 'w') as f:
        json.dump(scenes_struct, f, indent=2)
    return output_scene


def add_object(features: Features, args, camera):
    """
    Add random objects to the current blender scene
    """
    objects = []
    blender_objects = []

    # Choose a random size
    size_name, r = features.size
    obj_name, obj_name_out = features.shape
    color_name, rgba = features.color
    mat_name, mat_name_out = features.material
    x = features.x
    y = features.y

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
        r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * features.orientation

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)

    # Attach a random material
    utils.add_material(mat_name, Color=rgba)

    # Record dataset about the object in the scene dataset structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
        'shape': obj_name_out,
        'size': size_name,
        'material': mat_name_out,
        '3d_coords': tuple(obj.location),
        'rotation': theta,
        'pixel_coords': pixel_coords,
        'color': color_name,
    })

    return objects, blender_objects


def add_objects(features, args, camera):
    """
    Add random objects to the current blender scene
    """
    objects = []
    blender_objects = []
    for feature in features.values():

        # Choose a random size
        size_name, r = feature.size
        obj_name, obj_name_out = feature.shape
        color_name, rgba = feature.color
        mat_name, mat_name_out = feature.material
        x = feature.x
        y = feature.y

        # For cube, adjust the size a bit
        if obj_name == 'Cube':
            r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = 360.0 * feature.orientation

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)

        # Attach a random material
        utils.add_material(mat_name, Color=rgba)

        # Record dataset about the object in the scene dataset structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name_out,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where dataset[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in dataset['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the dataset image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix='.png')
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i + 1], p[i + 2], p[i + 3])
                          for i in range(0, len(p), 4))
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors: break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')
