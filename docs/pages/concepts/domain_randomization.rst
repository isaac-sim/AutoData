Domain Randomization
====================

Domain randomization adds variation to generated demonstrations by changing the environment's
appearance and initial state, resulting in a more diverse dataset. The top-level
``autodata_domain_randomization`` package provides functionality for randomizing lighting,
materials, camera parameters, and object and robot poses. Domain randomization settings can be
configured through the optional ``domain_randomization`` section of an
:doc:`environment profile <environment_profiles>`.

Configuration
-------------

Before creating an environment, AutoData applies ``domain_randomization`` settings only after
parsing the base task and applying the profile's scene and event changes. Randomizers can therefore
reference assets or events added by the same environment profile. Lighting, materials, and camera
parameter randomization are sampled at reset and are retained for the entire episode. Object poses
and robot root/joint positions are used to set the initial state, which may change during the
episode.

The table below lists all of the settings available directly under ``domain_randomization``.

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Setting
     - Parameters and constraints
   * - ``lighting``
     - Samples ``intensity_range`` and ``color_temperature_range`` [K] for the light specified by
       ``asset_name``. Modifying dome lights is not supported with parallel environments.
   * - ``poses``
     - Applies pose ranges through the reset event selected by ``event_name``. The event must accept
       ``pose_range``. ``position_xyz_min/max`` [m] and ``rpy_min/max`` [rad] follow that event's
       frame and absolute-position or offset semantics. Each entry replaces the event's entire
       ``pose_range``, including ranges set through ``events.override`` or ``events.add``. Other
       event parameters are preserved.
   * - ``cameras``
     - Randomizes the camera sensor selected by ``asset_name``. ``focal_length_scale_range`` acts as
       a multiplier on the nominal focal length and is not supported with parallel environments.
       ``position_xyz_min/max`` are displacements [m] from the nominal camera mounting position,
       measured along the axes of the link or scene node the camera is attached to.
       ``rotation_rpy_min_deg/max_deg`` are intrinsic XYZ rotation offsets [deg] from the nominal
       camera orientation, using optical axes: +X right, +Y down, +Z forward.
   * - ``material_colors``
     - Samples a color for the visuals selected by ``asset_name`` and ``prim_name_markers`` through
       the material's ``shader_input``. Colors are sampled between neighboring RGB entries in
       ``color_palette``, where each entry represents a fixed color. Depending on the shader, the
       input either sets the base color or tints the source texture. Tint multiplier values may
       exceed one. The selected prims must be non-instanceable.
   * - ``material_scalars``
     - Samples a value from ``value_range`` and writes it to the scalar ``shader_input``, for
       properties such as roughness or specular level. It uses the same visual selection and
       non-instanceable-prim requirement as ``material_colors``.
   * - ``robot_root``
     - Adds ``x_range_m`` and ``y_range_m`` offsets [m] and ``yaw_range_deg`` offsets [deg] to the
       default root of the ``robot`` articulation. This requires an Arena ``reset_all`` event whose
       function is ``isaaclab_arena.terms.events.reset_all_articulation_joints``.
   * - ``robot_joints``
     - Adds independent ``delta_range_deg`` offsets [deg] to matching ``joint_names`` or regex
       patterns around the nominal posture. Positions are clamped to joint limits and matching
       position targets are updated before controller initialization. The scene must contain an
       articulation named ``robot``, and its ``reset_all`` event must use Arena's
       ``reset_all_articulation_joints`` function.

.. important::

   All domain randomization sampling is done using Torch's shared RNGs, which are seeded from
   ``generation_policy.seed`` in the :doc:`task descriptor <task_descriptors>`.

   Scalar ranges use continuous uniform sampling, with independent draws for axes, joints, and
   environments. Pose ranges are passed to the existing reset event, which owns its sampler and
   constraints. When sampling material palettes, an adjacent anchor segment is uniformly chosen
   first. Then, a uniform interpolation fraction is sampled to determine the resulting color. Also
   note that subtask ``action_noise`` uses Gaussian noise and is configured separately in the
   :doc:`task descriptor <task_descriptors>`.

The following template configures every setting in the above table together.

.. code-block:: yaml

   name: all_randomization
   base_env: <compatible Arena environment id>
   description: All available domain randomization settings.
   domain_randomization:
     lighting:
       asset_name: task_light
       intensity_range: [250.0, 750.0]
       color_temperature_range: [3000.0, 8000.0]

     poses:
       - event_name: reset_object_pose
         position_xyz_min: [0.40, -0.10, 0.02]
         position_xyz_max: [0.60, 0.10, 0.02]
         rpy_min: [0.0, 0.0, -0.2]
         rpy_max: [0.0, 0.0, 0.2]

     cameras:
       - asset_name: robot_head_cam
         focal_length_scale_range: [0.98, 1.02]
         position_xyz_min: [-0.002, -0.002, -0.002]
         position_xyz_max: [0.002, 0.002, 0.002]
         rotation_rpy_min_deg: [-2.5, -2.5, -2.5]
         rotation_rpy_max_deg: [2.5, 2.5, 2.5]

     material_colors:
       - asset_name: object
         prim_name_markers: [mesh]
         shader_input: diffuseColor
         color_palette:
           - [0.8, 0.2, 0.2]
           - [0.2, 0.8, 0.2]
           - [0.2, 0.2, 0.8]

     material_scalars:
       - asset_name: object
         prim_name_markers: [mesh]
         shader_input: roughness
         value_range: [0.2, 0.8]

     robot_root:
       x_range_m: [-0.02, 0.02]
       y_range_m: [-0.02, 0.02]
       yaw_range_deg: [-5.0, 5.0]

     robot_joints:
       joint_names:
         - "(left|right)_shoulder_.*_joint"
         - "(left|right)_elbow_joint"
         - "(left|right)_wrist_.*_joint"
       delta_range_deg: [-5.0, 5.0]

Example Usage
-------------

The shipped :autodata_code_link:`<autodata_examples/env_profiles/g1_locomanip_dr.yaml>` profile
builds off of the environment from the
:doc:`humanoid workflow<../workflows/humanoid_dexmimicgen/index>`,
and randomizes the steering wheel's starting pose, rim color and roughness, and head-camera
calibration.

Run the code below to generate a small dataset from the annotated G1 dataset while applying
domain randomization.

.. code-block:: bash

   python scripts/generate_dataset.py \
       --env_name IsaacContrib-PickPlace-Locomanipulation-G1-Abs \
       --viz none \
       --device cpu \
       --num_envs 1 \
       --alg dexmimicgen \
       --generation_num_trials 5 \
       --task_descriptor autodata_examples/tasks/g1_pick_place.yaml \
       --embodiment autodata_examples/embodiments/g1_ik_abs.yaml \
       --env_profile autodata_examples/env_profiles/g1_locomanip_dr.yaml \
       --enable_cameras \
       --input_file ./datasets/annotated_datasets/dataset_g1_annotated.hdf5 \
       --output_file ./datasets/generated_dataset_g1_locomanip_dr.hdf5

.. note::

   This example randomizes camera intrinsics and requires ``--num_envs 1``. For parallel
   generation, omit ``focal_length_scale_range`` or set it to ``[1.0, 1.0]``. Also note that
   ``--enable_cameras`` is required to render and save camera observations in the dataset.
