Domain Randomization
====================

Domain randomization varies an environment's appearance and reset state while demonstrations
are generated. The top-level ``autodata_domain_randomization`` package owns the configuration
schema and the event functions for lighting, materials, camera calibration, and robot resets.
Configure waypoint noise through each subtask's ``action_noise`` in the
:doc:`task descriptor <task_descriptors>`.

Randomization is an optional ``domain_randomization`` section of the
:doc:`environment profile <environment_profiles>`. It stays separate from task and embodiment
YAML files, so the same task can be generated with different environment profiles. The schema lives in
:autodata_code_link:`<autodata_domain_randomization/configs.py>`; the shipped
:autodata_code_link:`<autodata_examples/env_profiles/franka_stack_randomized.yaml>` example
randomizes cube poses for Franka cube stacking. Its dome lighting is left unchanged.

Generating with Randomization
-----------------------------

Pass the environment profile containing the randomization settings with ``--env_profile``:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --env_name IsaacContrib-Stack-Cube-Franka-IK-Rel \
       --task_descriptor autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment autodata_examples/embodiments/franka_ik_rel.yaml \
       --env_profile autodata_examples/env_profiles/franka_stack_randomized.yaml \
       --input_file datasets/annotated_datasets/dataset_franka_annotated.hdf5 \
       --output_file datasets/randomized_franka.hdf5 \
       --generation_num_trials 10 --num_envs 2 --headless

This first example exercises reset poses. It omits lighting DR because the task uses dome
lighting, and it does not request image recording.
For a visual training dataset, use a task with configured cameras and pass ``--enable_cameras``;
see the G1 example below. A viewer window alone does not record images, and visual DR does not
add variation to state-only observations. The profile does not create missing camera sensors.

Configuration is applied in this order:

1. Parse the base task's environment configuration.
2. Apply ``--env_profile``, if supplied, including scene additions and reset-event changes.
3. Apply the profile's optional ``domain_randomization`` section to the resulting scene and events.
4. Create the environment and run generation. Reset events draw new episode settings.

A randomization configuration may therefore reference assets or reset events added by an
:doc:`environment profile <environment_profiles>`.

.. important::

   An enabled ``domain_randomization.poses`` entry replaces the **entire** ``pose_range`` of
   its named event. It takes precedence over the base environment's range and any range set
   through ``events.override`` or ``events.add``, because domain randomization is applied last.
   Individual axes are not merged with the earlier range; other event parameters are preserved.

Camera randomization requires ``--enable_cameras`` and a camera sensor already defined by the task.

Configuration and Lifecycle
---------------------------

Omitting ``domain_randomization`` leaves the base task's existing reset randomization and
waypoint noise unchanged. An empty mapping also makes no changes. Within the section, only the
supplied subsections are applied: a camera-only profile does not enable lighting randomization.

The presence of ``domain_randomization`` applies its supplied settings; there is no top-level
enable switch. Individual randomizers default to enabled when present and can use ``enabled: false``.
Those switches do not disable randomization already defined by the base task or the profile's ordinary events.

Lighting, materials, and camera calibration are sampled once per reset and retained for the episode.
Randomized object poses and robot root/joint positions set the initial state; they then evolve
under simulation. There are no ``episode``, ``scene``, or ``chunk`` wrappers under
``domain_randomization``.

The shipped profile uses the flat structure below:

.. literalinclude:: ../../../autodata_examples/env_profiles/franka_stack_randomized.yaml
   :language: yaml
   :start-at: name:

For a task exposing a ``robot_head_cam`` sensor, a camera-only profile can use the following
configuration. Replace ``base_env`` with the environment ID recorded in the source dataset:

.. code-block:: yaml

   name: camera_randomized
   base_env: <recorded environment id>
   domain_randomization:
     cameras:
       - asset_name: robot_head_cam
         focal_length_scale_range: [0.96, 1.04]
         position_xyz_min: [-0.005, -0.005, -0.005]
         position_xyz_max: [0.005, 0.005, 0.005]
         rotation_rpy_min_deg: [-3.0, -3.0, -3.0]
         rotation_rpy_max_deg: [3.0, 3.0, 3.0]

Asset and event names must match the selected task; they are not inferred from the filename.
The table lists settings within ``domain_randomization``.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Setting
     - Effect and parameters
   * - ``lighting``
     - Samples ``intensity_range`` and ``color_temperature_range`` [K] for the existing
       per-environment non-dome light named by ``asset_name``. Lighting DR is unsupported
       in environments containing dome lights, including single-environment runs.
   * - ``poses``
     - Overrides an existing ``event_name`` accepting ``pose_range``. Position bounds
       ``position_xyz_min/max`` are in [m]; ``rpy_min/max`` are in [rad], in the reset event's frame.
   * - ``cameras``
     - Samples a focal-length multiplier and mount offsets per camera and environment.
       Translation bounds are in the parent-link frame [m]; rotation bounds are intrinsic XYZ
       rotations about the camera's ROS optical axes [deg]. Offsets always use the nominal
       calibration, so repeated resets do not accumulate drift.
   * - ``material_colors``
     - Selects visual prims under ``asset_name`` using ``prim_name_markers`` and writes
       ``shader_input``. ``color_palette`` contains RGB multiplier anchors; samples interpolate
       between neighboring anchors. Multipliers may exceed one.
   * - ``material_scalars``
     - Uses the same asset/prim selection to sample a scalar ``shader_input`` uniformly from
       ``value_range``, for properties such as roughness or specular level.
   * - ``robot_root``
     - Adds ``x_range_m`` and ``y_range_m`` offsets [m] and ``yaw_range_deg`` offsets [deg] to
       the default robot root. Yaw is about world Z; height and nominal roll/pitch are preserved.
   * - ``robot_joints``
     - Adds independent ``delta_range_deg`` offsets [deg] to matching ``joint_names`` or regex
       patterns around the nominal reset posture. Positions are clamped to joint limits, and
       matching position targets are updated before controller initialization.

All Settings in One Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This template configures every setting in the table together. Replace ``base_env`` and the
asset, event, prim, shader, and joint selectors with names from your task; see
`Finding Asset, Event, and Shader Names`_. It assumes a compatible Arena task with an existing
per-environment non-dome light, a camera, and an object whose non-instanceable visual prims use
``UsdPreviewSurface``. It is not a drop-in profile for the native G1 example below.

.. code-block:: yaml

   name: all_randomization
   base_env: <compatible Arena environment id>
   domain_randomization:
     lighting:
       asset_name: task_light
       intensity_range: [250.0, 750.0]
       color_temperature_range: [3000.0, 8000.0]

     poses:
       - event_name: reset_object_pose
         # Position bounds follow this event's frame and pose semantics [m].
         position_xyz_min: [0.40, -0.10, 0.02]
         position_xyz_max: [0.60, 0.10, 0.02]
         rpy_min: [0.0, 0.0, -0.2]  # [rad]
         rpy_max: [0.0, 0.0, 0.2]

     cameras:
       - asset_name: robot_head_cam
         focal_length_scale_range: [0.98, 1.02]
         position_xyz_min: [-0.002, -0.002, -0.002]  # Parent-link frame [m].
         position_xyz_max: [0.002, 0.002, 0.002]
         rotation_rpy_min_deg: [-1.0, -1.0, -1.0]  # ROS optical axes [deg].
         rotation_rpy_max_deg: [1.0, 1.0, 1.0]

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
       yaw_range_deg: [-2.0, 2.0]

     robot_joints:
       joint_names:
         - "(left|right)_shoulder_.*_joint"
         - "(left|right)_elbow_joint"
         - "(left|right)_wrist_.*_joint"
       delta_range_deg: [-2.0, 2.0]

Lighting DR requires a scene without dome lights. The robot offsets require Arena's
``reset_all_articulation_joints`` function as the ``reset_all`` event, and the joint patterns
above target G1 arm joints. The ``poses`` entry replaces the selected event's entire
``pose_range``, as noted above. To try a subset, omit the other sections or set individual
randomizers to ``enabled: false``. Retain ``--enable_cameras`` when generating image observations.

G1 Visual Dataset Example
-------------------------

The native ``IsaacContrib-PickPlace-Locomanipulation-G1-Abs`` task used in the
:doc:`humanoid workflow <../workflows/humanoid_dexmimicgen/index>` exposes a dome named ``light``
and a camera sensor named ``robot_pov_cam``. Its recorded RGB observation is also named
``robot_pov_cam``. Arena G1 deployments may instead use ``robot_head_cam`` and
``robot_head_cam_rgb``; use the names from the task you actually launch.

Save the following complete profile as ``g1_visual_dr.yaml`` at the repository root:

.. code-block:: yaml

   name: g1_visual_dr
   base_env: IsaacContrib-PickPlace-Locomanipulation-G1-Abs
   description: Per-episode G1 head-camera calibration offsets.
   domain_randomization:
     cameras:
       - asset_name: robot_pov_cam
         focal_length_scale_range: [0.98, 1.02]
         position_xyz_min: [-0.002, -0.002, -0.002]
         position_xyz_max: [0.002, 0.002, 0.002]
         rotation_rpy_min_deg: [-1.0, -1.0, -1.0]
         rotation_rpy_max_deg: [1.0, 1.0, 1.0]

From an installed development environment with the repository's annotated G1 dataset available,
run a small image-recording pass:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --env_name IsaacContrib-PickPlace-Locomanipulation-G1-Abs \
       --alg dexmimicgen \
       --task_descriptor autodata_examples/tasks/g1_pick_place.yaml \
       --embodiment autodata_examples/embodiments/g1_ik_abs.yaml \
       --env_profile g1_visual_dr.yaml \
       --input_file datasets/annotated_datasets/dataset_g1_annotated.hdf5 \
       --output_file datasets/g1_visual_dr.hdf5 \
       --generation_num_trials 2 --num_envs 1 \
       --enable_cameras --device cpu --viz kit

The profile has been checked against the parsed task configuration. Live GPU rendering remains
unverified for this example. CPU physics is used here to match the humanoid workflow; camera
rendering still requires a supported GPU. On a
headless machine, replace ``--viz kit`` with ``--viz none`` while retaining ``--enable_cameras``.

This task uses dome lighting, so the profile deliberately omits ``lighting``. Lighting DR is
unsupported for this scene even with ``--num_envs 1``; its existing dome is left unchanged.
Camera DR can still be used in parallel environments.

This native G1 task uses Isaac Lab's ``reset_scene_to_default`` event, rather than Arena's
``reset_all_articulation_joints`` hook. Its visual DR profile therefore intentionally omits
``robot_root`` and ``robot_joints``. Those sections require a compatible Arena task even when
the robot itself is a G1.

Finding Asset, Event, and Shader Names
--------------------------------------

Start from the environment configuration registered for the chosen ``--env_name``. For the G1
example, inspect the module
``isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg`` in the Isaac Lab
submodule. The identifiers below belong to different namespaces:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Configuration field
     - Where to find its value
   * - ``asset_name``
     - The attribute under the scene configuration, such as ``light``, ``object``, or
       ``robot_pov_cam``. Use that scene name rather than the USD prim path.
   * - ``event_name``
     - An existing event term under the environment's event configuration. Its function must
       accept ``pose_range``. Pose values follow that function's semantics: some events take
       offsets from a default pose, while others use absolute positions in the environment.
   * - ``joint_names``
     - Joint names on the ``robot`` articulation, or regular expressions matching them.
       In an initialized environment, ``env.scene["robot"].joint_names`` lists the names and
       ``env.scene["robot"].find_joints(patterns)`` shows what a pattern selects.
   * - ``prim_name_markers``
     - Substrings of visual prim names under the target scene asset. Inspect the asset in
       Kit's Stage tree. Matching is case-sensitive and uses substrings, rather than regular
       expressions or full paths.
   * - ``shader_input``
     - An existing input on the material shader bound to a selected visual prim. Inspect its
       material binding and Shader properties in Kit. For a USD property named
       ``inputs:roughness``, configure ``roughness``. Shader names vary by asset and material model.
   * - Recorded image key
     - The observation term under the task's policy or camera observation group, then the
       corresponding dataset under ``data/demo_*/obs``. It can differ from the sensor name;
       inspect the source HDF5 instead of inferring a suffix.

Material Color and Scalar Examples
----------------------------------

The following block is a template to merge into an existing environment profile. It assumes a
scene asset named ``object`` with non-instanceable visual prims containing ``mesh`` in their
names, bound to a ``UsdPreviewSurface`` shader with ``diffuseColor`` and ``roughness`` inputs.
Replace those selectors with the values found on your asset before enabling it; this is not a
material preset for the native G1 steering-wheel asset.

.. code-block:: yaml

   domain_randomization:
     material_colors:
       - asset_name: object
         prim_name_markers: [mesh]
         shader_input: diffuseColor
         color_palette:
           - [0.8, 0.2, 0.2]
           - [0.2, 0.2, 0.8]
     material_scalars:
       - asset_name: object
         prim_name_markers: [mesh]
         shader_input: roughness
         value_range: [0.2, 0.8]

For this ``UsdPreviewSurface`` example, RGB values set the base diffuse color. A tint or
multiplier input on another shader interprets the sampled values according to that shader.
Use ``color_palette`` with one RGB entry to pin a color and equal ``value_range`` bounds to
pin a scalar. Inputs such as ``diffuse_tint`` or ``base_color_factor`` belong to other shaders;
substitute only inputs that actually exist. A missing matching input raises an error rather
than adding a new shader property. No lighting randomization is enabled by this block.

Robot Root and Initial Joint Example for Arena G1
-------------------------------------------------

For an Arena G1 task whose ``reset_all`` function is
``isaaclab_arena.terms.events.reset_all_articulation_joints``, merge this block into its profile.
Keep that task's own ``base_env`` and registration arguments. This block cannot be added to the
native G1 visual example above without changing its unsupported reset integration.

.. code-block:: yaml

   domain_randomization:
     robot_root:
       x_range_m: [-0.02, 0.02]
       y_range_m: [-0.02, 0.02]
       yaw_range_deg: [-2.0, 2.0]
     robot_joints:
       joint_names:
         - "(left|right)_shoulder_.*_joint"
         - "(left|right)_elbow_joint"
         - "(left|right)_wrist_.*_joint"
       delta_range_deg: [-2.0, 2.0]

These patterns select the G1 shoulder, elbow, and wrist joints. Confirm the matched names on
your articulation; this example leaves leg and hand joints at their existing reset values.
Joint offsets are independent per selected joint and environment, relative to the nominal
posture. Start with small ranges and check balance, reachability, controller initialization,
and task success before increasing them. The root's height and nominal roll/pitch are preserved.
The reset-hook and simulator-validation limitations below still apply.

Sampling Distributions
----------------------

Built-in DR draws use uniform distributions over the configured parameters. Continuous scalar
ranges use Torch's uniform sampler; independent axes, joints, and environments receive separate
draws where the randomizer supports them. Equal bounds pin a parameter to a fixed value.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Setting
     - Distribution
   * - ``lighting``
     - Uniform intensity and color temperature [K]. RGB light color is derived from the
       sampled temperature, so RGB components are not sampled independently.
   * - ``material_scalars``
     - Uniform values within each ``value_range``.
   * - ``cameras``
     - Uniform focal-length scale and independent uniform translation and intrinsic XYZ
       rotation components. Uniform Euler angles do not give a uniform distribution over
       all 3D orientations; uniform focal-length scales do not give uniform angular fields of view.
   * - ``robot_root``, ``robot_joints``
     - Independent uniform offsets within each configured range. Joint positions are then
       clamped to physical limits, which can concentrate samples at those limits.
   * - ``material_colors``
     - Uniform choice among adjacent palette segments, followed by a uniform interpolation
       fraction within the chosen segment. This samples the palette path, rather than each RGB
       channel independently; unequal segment lengths do not receive length-proportional weights.
       A one-entry palette pins the color.
   * - ``poses``
     - Passes ranges to the existing task reset event. That event owns its sampling and any
       constraints, such as minimum object separation; the DR profile does not replace its sampler.

These rules describe domain randomization. Subtask ``action_noise`` uses independent Gaussian
noise, configured separately in the :doc:`task descriptor <task_descriptors>`.

Parallel Environments and Asset Constraints
-------------------------------------------

* **Lighting DR is unsupported in environments with dome lighting.** Dome lights illuminate
  the whole scene; requesting lighting DR in such a scene raises a clear error on the first
  randomization call. Omit ``domain_randomization.lighting`` to use the other randomizers
  while leaving the dome unchanged.
* Lighting DR requires one existing non-dome light per environment, under that environment's
  prim path. It edits intensity and color only. The task remains responsible for light placement
  and linking to avoid illumination spilling across environments. No lights, enclosures, or
  spacing changes are added by DR.
* Material randomization requires non-instanceable target visual prims. Per-asset material
  overrides isolate shared source materials, so a partial reset does not alter other environments.
  One material draw is shared by the environments in a reset batch; camera and supported lighting
  draws are independent per environment.
* Robot root/joint settings require an Arena ``reset_all`` event whose function is
  ``reset_all_articulation_joints`` and an articulation named ``robot``. AutoData replaces that
  reset locally and rejects other reset functions. Being an Arena task alone does not establish
  compatibility. For example, the
  :autodata_code_link:`<autodata_examples/envs/isaac_lab_arena/franka_rope/embodiment.py>`
  Franka rope task uses ``reset_scene_to_default``, so these robot DR sections are currently
  unsupported for that task. Its existing base-task reset randomization is unchanged.
* Camera randomization preserves the nominal principal point and calibrated OpenCV focal-axis
  relationship where available. RTX camera render products are made independent to avoid tiled
  rendering reusing another environment's projection.

.. note::

   Robot root/joint DR is not restricted by robot type, but the reset-hook requirement above
   limits which tasks can use it. The robot-offset tests use mock articulations; fixed-base
   root relocation and backend-specific anchoring have not been validated in a live simulator.
   Fixed-base manipulator root DR should therefore be treated as unverified, even when its
   task has the required reset hook. CPU checks also do not establish live GPU/RTX rendering
   correctness for camera or lighting changes.

Generation uses the task descriptor's seed for the Torch RNG draws. With ``--record_metric_inputs``
and ``--result_file``, the result JSON records the environment profile's path and SHA-256 hash
in ``run.inputs.env_profile``, including its randomization settings; see :doc:`data_generator`.
That hash identifies the configured ranges, rather than every episode's sampled camera values.

Checking the Recorded Result
----------------------------

1. Generate two episodes with one environment before scaling up. Confirm that the run finishes
   and that the requested images exist, then run the structural validator:

   .. code-block:: bash

      python scripts/validate_dataset.py datasets/g1_visual_dr.hdf5

2. Inspect the recorded observations. With ``h5py`` and Pillow available, the following script
   prints the image layout and saves the first RGB frame from each of the first two G1 episodes:

   .. code-block:: python

      import h5py
      from PIL import Image

      with h5py.File("datasets/g1_visual_dr.hdf5", "r") as dataset:
          names = sorted(dataset["data"], key=lambda name: int(name.rsplit("_", 1)[-1]))
          for name in names[:2]:
              observations = dataset["data"][name]["obs"]
              print(name, list(observations))
              frames = observations["robot_pov_cam"]
              print(frames.shape, frames.dtype)
              Image.fromarray(frames[0, ..., :3]).save(f"{name}_first.png")

   Image arrays should contain one RGB image per recorded frame. If ``robot_pov_cam`` is missing,
   check ``--enable_cameras`` and the task's configured image observations. A visible Kit window
   does not prove that images were exported.

3. Compare the saved frames and inspect an episode's image sequence. Configured visual
   randomization draws should vary between episodes and remain consistent within an episode; images still
   change as the camera and scene move. For root/joint DR, also check the recorded initial state
   and ensure the robot settles and completes the task. Inspect exported images for visual DR:
   replaying only the recorded physical state does not reconstruct an unrecorded lighting draw.
4. Establish a pinned calibration baseline before increasing camera ranges: set
   ``focal_length_scale_range: [1.0, 1.0]`` and all camera translation/rotation bounds to zero.
   Zero focal-length scale is invalid. Pin lighting and materials to their known nominal values
   when comparing against an unrandomized source. Then vary one setting at a time.
5. Increase parallelism only after this small run looks correct, and watch rendering memory,
   storage, and generation success rate.

Re-rendering Recorded Datasets
------------------------------

:autodata_code_link:`<scripts/rerender_camera_dr.py>` applies visual randomization to an existing
image dataset. It injects recorded states and renders a new RGB observation while copying actions,
states, and other observations into a separate output file. Pose and robot-root/joint
randomization are ignored during replay, so the recorded trajectory is retained.

Use a profile that reproduces the source scene and adds the desired visual randomization,
along with the registration options from the source generation run. For a dataset
with the ``robot_head_cam_rgb`` observation and the camera-only YAML above saved as
``camera_profile.yaml``:

.. code-block:: bash

   python scripts/rerender_camera_dr.py \
       --input_file datasets/with_images.hdf5 \
       --output_file datasets/randomized_images.hdf5 \
       --env_profile camera_profile.yaml \
       --camera_key robot_head_cam_rgb --num_envs 2 --headless

.. note::

   The current re-render script requires ``--camera_key`` to end in ``_rgb`` and derives
   the sensor name by removing that suffix. For example, ``robot_head_cam_rgb`` maps to
   ``robot_head_cam``. The native G1 example above records ``robot_pov_cam`` without that
   suffix, so its dataset cannot currently use this script unchanged. Passing an invented
   ``robot_pov_cam_rgb`` key does not rename the recorded observation. Generation with
   visual DR remains available for that G1 task.

The environment ID defaults to the one recorded in the source dataset; ``--env_name`` can
override it. Cameras are enabled by the script. The output camera resolution and dtype must
match the source. Rendering requires Isaac Sim and a supported GPU.

* ``--validate_episodes N`` renders a subset without writing an output dataset. It reports
  sampled camera calibration, image differences, and temporal alignment. Use the pinned
  calibration baseline above to check alignment; large visual changes or insufficient motion
  can make it inconclusive.
* Each episode is seeded with ``--dr_seed + episode index``. Numeric episode order keeps the
  draw independent of batch size and resume position. Camera calibration values and the episode
  seed are stored as output episode attributes.
* ``--resume`` skips completed episodes and discards incomplete image writes. It requires the
  same source path, environment-profile contents, camera key, and seed. An existing output file is
  otherwise left intact rather than overwritten.

For Python callers, the package exports ``DomainRandomizationCfg``,
``create_domain_randomization_profile()``, and ``apply_domain_randomization()`` from
:autodata_code_link:`<autodata_domain_randomization/__init__.py>`.
