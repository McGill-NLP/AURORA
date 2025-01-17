# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Worker file for the Multi-Object Video (MOVi) C (and CC) datasets.
  * The number of objects is randomly chosen between
    --min_num_objects (3) and --max_num_objects (10)
  * The objects are randomly chosen from the Google Scanned Objects dataset

  * Background is an random HDRI from the HDRI Haven dataset,
    projected onto a Dome (half-sphere).
    The HDRI is also used for lighting the scene.
"""

import logging

import bpy
import copy
import os
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import random
import shutil

from GSO_transfer import GSO_dict
from utils import save_scene_instruction, dataset_dir

# --- Some configuration values
DATASET_TYPE = "closer"
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-8, -8, 0), (8, 8, 5)]
SPAWN_REGION_OBJ = [[-6, -6, 0.5], [6, 6, 0.5]]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_objects", type=int, default=1,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=5,
                    help="maximum number of objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                    default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=4.0)
parser.add_argument("--smallest_scale", type=float, default=2.)
parser.add_argument("--largest_scale", type=float, default=4.)

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                    resolution=512)
parser.add_argument("--sub_outputdir", type=str, default="test sub output dir")
parser.add_argument("--generate_idx", type=int, default=-1, help="generation idx")
FLAGS = parser.parse_args()


import pyquaternion as pyquat
def default_rng():
  return np.random.RandomState()

def random_rotation(axis=None, rng=default_rng()):
  """ Compute a random rotation as a quaternion.
  If axis is None the rotation is sampled uniformly over all possible orientations.
  Otherwise it corresponds to a random rotation around the given axis."""

  if axis is None:
    # uniform across rotation space
    # copied from pyquat.Quaternion.random to be able to use a custom rng
    r1, r2, r3 = rng.random(3)

    q1 = np.sqrt(1.0 - r1) * (np.sin(2 * np.pi * r2))
    q2 = np.sqrt(1.0 - r1) * (np.cos(2 * np.pi * r2))
    q3 = np.sqrt(r1) * (np.sin(2 * np.pi * r3))
    q4 = np.sqrt(r1) * (np.cos(2 * np.pi * r3))

    return q1, q2, q3, q4

  else:
    if isinstance(axis, str) and axis.upper() in ["X", "Y", "Z"]:
      axis = {"X": (1., 0., 0.),
              "Y": (0., 1., 0.),
              "Z": (0., 0., 1.)}[axis.upper()]

    # quat = pyquat.Quaternion(axis=axis, angle=rng.uniform(0, 2*np.pi))
    quat = pyquat.Quaternion(axis=axis, angle=rng.uniform(-0.5*np.pi, 0.5*np.pi)) # -0.5pi -- 0.5pi
    return tuple(quat)


from kubric.core import objects
def rotation_sampler(axis=None):
  def _sampler(obj: objects.PhysicalObject, rng):
    obj.quaternion = random_rotation(axis=axis, rng=rng)
  return _sampler


def move_until_no_overlap(asset, simulator, spawn_region=((-1, -1, -1), (1, 1, 1)), max_trials=100,
                          rng=default_rng()):
  return kb.randomness.resample_while(asset,
                        # samplers=[rotation_sampler(axis='Z'), kb.randomness.position_sampler(spawn_region)],
                        samplers=[kb.randomness.position_sampler(spawn_region)],
                        condition=simulator.check_overlap,
                        max_trials=max_trials,
                        rng=rng)


def check_ok(obj, pos, region):
    # import pdb; pdb.set_trace()
    x, y, z = pos
    if pos[0]<region[0][0] or pos[0]>region[1][0] or pos[1]<region[0][1] or pos[1]>region[1][1]: #or pos[2]<region[0][2] or  pos[2]>region[1][2]:
        return False

    if simulator.check_overlap(obj):
        return False

    return True


def get_obj_x_left(bound, scale):
    return -bound[0][0] * scale[0]

def get_obj_x_right(bound, scale):
    return bound[1][0] * scale[0]

def get_obj_y_front(bound, scale):
    return -bound[0][1] * scale[1]

def get_obj_y_behind(bound, scale):
    return bound[1][1] * scale[1]

def get_obj_z(bound, scale):
    return bound[0][2] * scale[2]

def get_obj_z_up(bound, scale):
    return bound[1][2] * scale[2]


# def get_new_pos(bounds, scale, ref_location, ref_pos, ref_z_up, ref_object,  rng):
#     obj_z = - get_obj_z(bounds, scale)
#     # import pdb; pdb.set_trace()
#     ref_x_left, ref_x_right, ref_y_front, ref_y_behind = get_obj_x_left(ref_object.bounds, ref_object.scale), get_obj_x_right(ref_object.bounds, ref_object.scale), get_obj_y_front(ref_object.bounds, ref_object.scale), get_obj_y_behind(ref_object.bounds, ref_object.scale)
#     ref_x, ref_y, ref_z = ref_pos
#     if ref_location == 'front':
#         return [rng.uniform(ref_x-0.5, ref_x+0.5), rng.uniform(ref_y-ref_y_front-6, ref_y-ref_y_front-2), obj_z+0.02]
#     elif ref_location == 'behind':
#         return [rng.uniform(ref_x-0.5, ref_x+0.5), rng.uniform(ref_y+ref_y_behind+2, ref_y+ref_y_behind+6), obj_z+0.02]
#     elif ref_location == 'left':
#         return [rng.uniform(ref_x-ref_x_left-6, ref_x-ref_x_left-2), rng.uniform(ref_y-0.5, ref_y+0.5), obj_z+0.02]
#     elif ref_location == 'right':
#         return [rng.uniform(ref_x+ref_x_right+2, ref_x+ref_x_right+6), rng.uniform(ref_y-0.5, ref_y+0.5), obj_z+0.02]
#     elif ref_location == 'on':
#         return [ref_x, ref_y, ref_z+ref_z_up+obj_z+1]

def get_new_pos(bounds, scale, ref_location, ref_pos, ref_z_up, ref_object, rng):
    # Calculate the z-position based on the object's bounds and scale
    obj_z = -get_obj_z(bounds, scale) + 0.2  # Ensuring the object is slightly above the ground

    # Extract the bounds from the SPAWN_REGION_OBJ variable
    spawn_x_min, spawn_y_min, _ = SPAWN_REGION_OBJ[0]
    spawn_x_max, spawn_y_max, _ = SPAWN_REGION_OBJ[1]

    # Calculate the maximum offsets within the given spawn bounds
    max_offset_x = spawn_x_max - spawn_x_min + 1
    max_offset_y = spawn_y_max - spawn_y_min + 1

    # Define absolute positions for each location using straightforward calculations
    import random

    # Return the position for the specified location
    return locations.get(ref_location, [ref_pos[0], ref_pos[1], obj_z])  # Default position if location is not specified

def add_new_obj(scene, new_obj, ref_location, ref_object, rng, max_trails=50):

    ref_obj_pos = ref_object.position
    # import pdb; pdb.set_trace()
    ref_obj_z_up = get_obj_z_up(ref_object.bounds, ref_object.scale)
    new_obj_pos = get_new_pos(new_obj.bounds, new_obj.scale, ref_location, ref_obj_pos, ref_obj_z_up, ref_object, rng)
    new_obj.position = new_obj_pos
    scene += new_obj

    # import pdb; pdb.set_trace()
    trails = 0
    while not check_ok(new_obj, new_obj.position, SPAWN_REGION_OBJ):
        trails += 1
        # import pdb; pdb.set_trace()
        new_obj.position = get_new_pos(new_obj.bounds, new_obj.scale, ref_location, ref_obj_pos, ref_obj_z_up, ref_object, rng)
        # new_obj.quaternion = random_rotation(axis="Z", rng=rng)
        if trails > max_trails:
            print('cannot put the object, break')
            # import pdb; pdb.set_trace()
            return None
    print('try {} times'.format(trails))
    return scene

def gen_caption(obj_name, obj_scale, ref_obj_name, ref_obj_scale, type='closer'):
    if type == 'closer':
        captions = [f'Move the {obj_name} and the {ref_obj_name} closer together.', f'Move the {obj_name} and the {ref_obj_name} closer.', f'Move the {obj_name} and the {ref_obj_name} closer to each other.', f'Move both objects closer', f'Move the two objects closer together.', f'Move the two objects closer to each other.']
    elif type == 'further':
        captions = [f'Move the {obj_name} further away from the {ref_obj_name}.', f'Move the {obj_name} and the {ref_obj_name} further apart.', f'Move the {obj_name} and the {ref_obj_name} further away from each other.', f'Move both objects further apart', f'Move the two objects further away from each other.', f'Move the two objects further apart.']
    elif type == 'swap':
        captions = [f'Swap the positions of the {obj_name} and the {ref_obj_name}.', f'Swap the positions of the {ref_obj_name} and the {obj_name}.', f'Swap the positions of the two objects.', f'Swap positions of both items.']
    return random.choice(captions)


# --- Common setups & resources
print('Generate {} Sample'.format(FLAGS.generate_idx))
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
output_dir = output_dir / FLAGS.sub_outputdir



simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


# --- Populate the scene
# background HDRI
train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.)
logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
hdri_id = rng.choice(train_backgrounds)

background_hdri = hdri_source.create(asset_id=hdri_id)
#assert isinstance(background_hdri, kb.Texture)
logging.info("Using background %s", hdri_id)
scene.metadata["background"] = hdri_id
renderer._set_ambient_light_hdri(background_hdri.filename)


# Dome
dome = kubasic.create(asset_id="dome", name="dome",
                      friction=FLAGS.floor_friction,
                      restitution=FLAGS.floor_restitution,
                      static=True, background=True)
assert isinstance(dome, kb.FileBasedObject)
scene += dome
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)



def get_linear_camera_motion_start_end(
    movement_speed: float,
    inner_radius: float = 8.,
    outer_radius: float = 12.,
    z_offset: float = 0.1,
):
  """Sample a linear path which starts and ends within a half-sphere shell."""
  while True:
    camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                 outer_radius,
                                                                 z_offset))
    direction = rng.rand(3) - 0.5
    movement = direction / np.linalg.norm(direction) * movement_speed
    camera_end = camera_start + movement
    if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
        camera_end[2] > z_offset):
      return camera_start, camera_end


# Camera
logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=36)
if FLAGS.camera == "fixed_random":
    # scene.camera.position = kb.sample_point_in_half_sphere_shell(
    #     inner_radius=7., outer_radius=9., offset=4)
    scene.camera.position = (0, -10, 15)
    scene.camera.look_at((0, 0, 0))
elif FLAGS.camera == "linear_movement":
    camera_start, camera_end = get_linear_camera_motion_start_end(
      movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement)
    )
    # linearly interpolate the camera position between these two points
    # while keeping it focused on the center of the scene
    # we start one frame early and end one frame late to ensure that
    # forward and backward flow are still consistent for the last and first frames
    for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
        interp = ((frame - FLAGS.frame_start + 1) /
                  (FLAGS.frame_end - FLAGS.frame_start + 3))
        scene.camera.position = (interp * np.array(camera_start) +
                                 (1 - interp) * np.array(camera_end))
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)


# Add random objects
train_split, test_split = gso.get_test_split(fraction=0.)
# if FLAGS.objects_split == "train":
logging.info("Choosing one of the %d training objects...", len(train_split))
# active_split = train_split
active_split = list(GSO_dict.keys())

# num_objects = rng.randint(FLAGS.min_num_objects,
                        #   FLAGS.max_num_objects+1)
num_objects = 2

logging.info("Step 1: Randomly placing %d objects:", num_objects)
object_state_save_dict = {}
object_state_ref_dict = {}


# not resample objects
object_id_list = random.sample(active_split, num_objects+1)

for i in range(num_objects):
    # object_id = rng.choice(active_split)
    object_id = object_id_list[i]
    obj = gso.create(asset_id=object_id)


    assert isinstance(obj, kb.FileBasedObject)
    scale = rng.uniform(FLAGS.smallest_scale, FLAGS.largest_scale)
    obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])


    obj_pos_z = - get_obj_z(obj.bounds, obj.scale)
    SPAWN_REGION_OBJ[0][2], SPAWN_REGION_OBJ[1][2] = obj_pos_z, obj_pos_z
    obj.position = rng.uniform(*SPAWN_REGION_OBJ)

    obj.metadata["scale"] = scale
    scene += obj
    move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_OBJ, rng=rng)
    # initialize velocity randomly but biased towards center
    # obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
    #                 [obj.position[0], obj.position[1], 0])
    # print(obj.position)
    obj.velocity = [0, 0, 0]
    logging.info("    Added %s at %s", obj.asset_id, obj.position)
    object_state_save_dict[i] = {'object_id': object_id,
                               'object_scale': obj.scale,
                               'object_quaternion': obj.quaternion,
                               'object_bounds': obj.bounds}
    object_state_ref_dict[i] = {'object': obj}


ref_object = object_state_ref_dict[list(object_state_ref_dict.keys())[0]]['object']
ref_object_name = GSO_dict[ref_object.asset_id]
ref_location = ref_object.position

obj = object_state_ref_dict[list(object_state_ref_dict.keys())[1]]['object']
new_object_name = GSO_dict[obj.asset_id]

# 1st
print('Generate the first scene.')
# object_id = rng.choice(active_split)
# object_id = object_id_list[-1]
# obj = gso.create(asset_id=object_id)
# scale = rng.uniform(FLAGS.smallest_scale, FLAGS.largest_scale)
# obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
# obj.metadata["scale"] = scale

# new_object_name = GSO_dict[obj.asset_id]
# print('Add new object {}'.format(new_object_name))


# scene = add_new_obj(scene, obj, ref_location1, ref_object, rng, max_trails=500)
if scene is None:
    exit()
frame = renderer.render_still()

edits = ['close', 'swap']
edit = random.choice(edits)

os.makedirs(output_dir/'{}'.format(FLAGS.generate_idx), exist_ok=True)
kb.write_png(frame["rgba"], output_dir/"{}/image0.png".format(FLAGS.generate_idx))
caption_1 = gen_caption(new_object_name, obj.metadata["scale"], ref_object_name, ref_object.metadata["scale"], type="further" if edit == 'close' else 'swap')
print(caption_1)


# save meta ann
object_state_save_dict[i+1] = {'object_id': object_id,
                               'object_scale': obj.scale,
                               'object_pos': obj.position,
                               'object_quaternion': obj.quaternion,
                               'object_bounds': obj.bounds}
# import json
# json.dump(object_state_save_dict, open(output_dir/'{}/meta_ann1.json'.format(FLAGS.generate_idx), 'w'))
# np.save(output_dir/'{}/meta_ann1.npy'.format(FLAGS.generate_idx), object_state_save_dict)
# renderer.save_state(output_dir/'{}/image1.blend'.format(FLAGS.generate_idx))



# 2nd
print('Generate the second scene.')
# delete the last object to generate the second frame
# import pdb; pdb.set_trace()
# scene.remove(obj)
# scene= add_new_obj(scene, obj, ref_location2, ref_object, rng, max_trails=100)
ref_obj_pos = ref_object.position
ref_obj_z_up = get_obj_z_up(ref_object.bounds, ref_object.scale)
logging.info(f'Object position: {obj.position}')

obj1_pos = obj.position
obj2_pos = ref_location
if edit == 'close':
    direction = obj2_pos - obj1_pos
    factor1, factor2 = random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)
    obj.position = obj1_pos + direction * factor1
    ref_object.position = obj2_pos - direction * factor2
else:
    obj.position = obj2_pos
    ref_object.position = obj1_pos

frame = renderer.render_still()
kb.write_png(frame["rgba"], output_dir/"{}/image1.png".format(FLAGS.generate_idx))
caption_2 = gen_caption(new_object_name, obj.metadata["scale"], ref_object_name, ref_object.metadata["scale"], type="closer" if edit == 'close' else 'swap')
print(caption_2)

# save meta ann
object_state_save_dict[i+1] = {'object_id': object_id,
                               'object_scale': obj.scale,
                               'object_pos': obj.position,
                               'object_quaternion': obj.quaternion,
                               'object_bounds': obj.bounds}
# import json
# json.dump(object_state_save_dict, open(output_dir/'{}/meta_ann2.json'.format(FLAGS.generate_idx), 'w'))
# np.save(output_dir/'{}/meta_ann2.npy'.format(FLAGS.generate_idx), object_state_save_dict)
# renderer.save_state(output_dir/'{}/image2.blend'.format(FLAGS.generate_idx))


# save json
# local_ann = {'image0':"{}/image0.png".format(FLAGS.generate_idx), 'caption0':caption_1,
#              'image1':"{}/image1.png".format(FLAGS.generate_idx), 'caption1':caption_2,
#              'ann_path':"{}/ann.json".format(FLAGS.generate_idx),
#              'obj_num':num_objects+1}
# json.dump(local_ann, open("{}/{}/ann.json".format(str(output_dir), FLAGS.generate_idx), 'w'))

# import pdb; pdb.set_trace()
# if not os.path.exists("{}/global_ann.json".format(str(output_dir))):
#     json.dump([], open("{}/global_ann.json".format(str(output_dir)), 'w'))
# with open("{}/global_ann.json".format(str(output_dir)), 'r') as f:
#     old_data = json.load(f)
#     old_data.append(local_ann)
# with open("{}/global_ann.json".format(str(output_dir)), "w") as f:
#     json.dump(old_data, f)

local_ann = [{
        'input': dataset_dir(DATASET_TYPE) + "{}/image0.png".format(FLAGS.generate_idx), 
        'output': dataset_dir(DATASET_TYPE) + "{}/image1.png".format(FLAGS.generate_idx), 
        'instruction': caption_2,
    },
    {
        'input': dataset_dir(DATASET_TYPE) + "{}/image1.png".format(FLAGS.generate_idx), 
        'output': dataset_dir(DATASET_TYPE) + "{}/image0.png".format(FLAGS.generate_idx), 
        'instruction': caption_1,
    }
]
save_scene_instruction(f"{output_dir}/eq_kubric_{DATASET_TYPE}.json", local_ann, DATASET_TYPE, FLAGS.generate_idx)

kb.done()
