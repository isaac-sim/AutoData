..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Migration from Isaac Lab Mimic to AutoData: Franka Cube Stacking
================================================================

This guide migrates the Franka cube-stacking task from Isaac Lab Mimic to AutoData. Every step
uses the completed Franka files and commands in this repository. Apply the same mapping to the
corresponding files when migrating another Isaac Lab Mimic task.

AutoData uses standard Isaac Lab ManagerBasedRLEnv environments. For Franka cube stacking, keep the normal Isaac Lab
environment and replace the Mimic environment ID
``Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`` with
``Isaac-Stack-Cube-Franka-IK-Rel-v0``. The Mimic-specific configuration moves into AutoData's
task and embodiment descriptors.

What moves where?
-----------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Isaac Lab Mimic
     - Isaac AutoData
   * - Normal scene, actions, reset events, observations, and success condition
     - Remain in the normal Isaac Lab environment config
   * - ``MimicEnvCfg.datagen_config``
     - ``generation_policy`` in the task descriptor YAML
   * - ``MimicEnvCfg.subtask_configs``
     - ``subtasks`` in the task descriptor YAML
   * - Mimic environment methods for reading EEF poses and converting actions
     - Embodiment YAML and its registered embodiment adapter
   * - Mimic environment methods for reading object poses and subtask signals
     - AutoData's ``Datastream`` reads them from the normal environment
   * - Isaac Lab Mimic generation scripts
     - AutoData's ``annotate_demos.py`` and ``generate_dataset.py``

AutoData runs on standard Isaac Lab ``ManagerBasedRLEnv`` environments. It does not require
``ManagerBasedRLMimicEnv``, ``MimicEnvCfg``, or any other import from ``isaaclab_mimic``.

Step 1: Use the normal Isaac Lab environment
--------------------------------------------

The Isaac Lab Mimic implementation of Franka cube stacking combines two elements:

* ``FrankaCubeStackEnvCfg`` defines the simulated task as a standard Isaac Lab ManagerBasedRLEnv environment.
* ``FrankaCubeStackIKRelMimicEnvCfg`` and ``FrankaCubeStackIKRelMimicEnv`` add data-generation
  configuration and adapter methods.

AutoData uses ``FrankaCubeStackEnvCfg`` directly. Change the environment ID as follows:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Isaac Lab Mimic
     - AutoData
   * - ``Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0``
     - ``Isaac-Stack-Cube-Franka-IK-Rel-v0``

For your own Isaac Lab environment, verify that it provides:

* a ``success`` termination term;
* scene object names matching the task descriptor's ``object_ref`` values;
* policy observations for the EEF position and quaternion; and
* for automatic annotation, a non-concatenated ``subtask_terms`` observation group containing
  the signals named by the task descriptor.

The Franka cube-stack environment already satisfies these requirements: its objects are
``cube_1``, ``cube_2``, and ``cube_3``; its EEF observations are ``eef_pos`` and ``eef_quat``; and
its automatic annotation signals are ``grasp_1``, ``stack_1``, and ``grasp_2``.

Step 2: Move Isaac Lab Mimic ManagerBasedRLMimicEnvCfg into an AutoData task descriptor
---------------------------------------------------------------------------------------

An Isaac Lab Mimic environment config contains two kinds of data-generation information:

* ``datagen_config`` contains settings for the complete generation run.
* ``subtask_configs`` describes the ordered object-relative segments that MimicGen transforms
  and stitches together.

An AutoData task descriptor stores the same information as data rather than Python code.

Franka cube-stacking conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Franka cube stacking, the Isaac Lab Mimic source is
``FrankaCubeStackIKRelMimicEnvCfg`` in
``isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py``. The completed result is
:isaac_autodata_code_link:`<isaac_autodata_examples/tasks/franka_cube_stack.yaml>`.

First, the run-wide Isaac Lab Mimic settings:

.. code-block:: python

   self.datagen_config.name = "demo_src_stack_isaac_lab_task_D0"
   self.datagen_config.generation_guarantee = True
   self.datagen_config.generation_keep_failed = False
   self.datagen_config.generation_num_trials = 10
   self.datagen_config.generation_select_src_per_subtask = True
   self.datagen_config.generation_transform_first_robot_pose = False
   self.datagen_config.generation_interpolate_from_last_target_pose = True
   self.datagen_config.seed = 1

become the AutoData task descriptor's ``generation_policy``:

.. code-block:: yaml

   name: franka_cube_stack
   description: Stack red, green, and blue cubes into a single tower.
   algo: mimicgen

   generation_policy:
     name: franka_cube_stack
     seed: 1
     num_trials: 10
     guarantee_success: true
     keep_failed: false
     select_src_per_subtask: true
     transform_first_robot_pose: false
     interpolate_from_last_target_pose: true

Next, the subtasks of Isaac Lab Mimic are moved. Isaac Lab Mimic's subtask structure is:

.. code-block:: python

   SubTaskConfig(
       object_ref="cube_2",
       subtask_term_signal="grasp_1",
       subtask_term_offset_range=(10, 20),
       selection_strategy="nearest_neighbor_object",
       selection_strategy_kwargs={"nn_k": 3},
       action_noise=0.03,
       num_interpolation_steps=5,
       num_fixed_steps=0,
       apply_noise_during_interpolation=False,
       description="Grasp red cube",
   )

In ``franka_cube_stack.yaml``, that same subtask is:

.. code-block:: yaml

   subtasks:
     franka:
       - object_ref: cube_2
         description: Grasp red cube.
         subtask_term_signal: grasp_1
         subtask_term_offset_range: [10, 20]
         selection_strategy: nearest_neighbor_object
         selection_strategy_kwargs: {nn_k: 3}
         action_noise: 0.03
         num_interpolation_steps: 5
         num_fixed_steps: 0
         apply_noise_during_interpolation: false

The other three ``SubTaskConfig`` objects for Franka cube stacking are converted identically in the completed YAML.

Step 3: Move the Isaac Lab Mimic ManagerBasedRLMimicEnv into an AutoData embodiment descriptor
----------------------------------------------------------------------------------------------

An Isaac Lab Mimic environment wrapper implements the robot-specific interface used during data
generation. Its methods define:

* Where to read each end-effector pose.
* How the environment's action vector encodes an end-effector target.
* Which action dimensions control the gripper or other non-pose channels.

In AutoData, this interface is provided by an embodiment adapter configured through an embodiment
descriptor.

Franka cube-stacking conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Franka cube stacking, the Isaac Lab Mimic source is
``FrankaCubeStackIKRelMimicEnv`` in
``isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py``. The completed result is
:isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/franka_ik_rel.yaml>`.

First, the Isaac Lab Mimic ManagerBasedRLMimicEnv reads the Franka end-effector pose from two observations in the
``policy`` group:

.. code-block:: python

   eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
   eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]

The embodiment descriptor in AutoData records those observation keys and uses the same ``franka`` EEF name
as the task descriptor:

.. code-block:: yaml

   eef_name: franka

   pose_obs_keys:
     pos: eef_pos
     quat: eef_quat

Next, the Isaac Lab Mimic ManagerBasedRLMimicEnv action conversion methods show that the environment uses a relative
pose action:

.. code-block:: python

   delta_position = target_pos - curr_pos
   delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
   delta_rotation = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(delta_rot_mat))
   pose_action = torch.cat([delta_position, delta_rotation], dim=0)

   return torch.cat([pose_action, gripper_action], dim=0)

The reverse conversion reads position from ``action[:, :3]`` and compact axis-angle rotation from
``action[:, 3:6]``. ``actions_to_gripper_actions()`` reads ``action[:, -1:]``, so the final action
dimension is the gripper command. This is exactly the action convention implemented by AutoData's
``delta_pose_ik_single_arm`` adapter:

.. code-block:: yaml

   type: delta_pose_ik_single_arm

   action_layout:
     gripper_dim: 1
     clip_pose_action_to_unit: true

The embodiment adapter now provides the pose reads and action conversions, while the Datastream
exposes them to the generator. The corresponding methods are no longer needed on a Mimic
environment wrapper.

Putting the pieces together, the complete AutoData Franka embodiment descriptor is:

.. code-block:: yaml

   type: delta_pose_ik_single_arm
   name: franka_panda
   description: Franka 7-DOF arm with parallel gripper, delta-pose IK control.

   eef_name: franka

   pose_obs_keys:
     pos: eef_pos
     quat: eef_quat

   action_layout:
     gripper_dim: 1
     clip_pose_action_to_unit: true

   eef_offset: [0.0, 0.0, 0.0]


Step 4: Reuse or annotate source demonstrations
-----------------------------------------------

The Franka cube stacking source demonstrations recorded in Isaac Lab Mimic are already in the
correct HDF5 format. Annotate the raw dataset directly with AutoData:

.. code-block:: bash

   python scripts/annotate_demos.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --viz none \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/dataset_franka.hdf5 \
       --output_file ./datasets/dataset_franka_annotated.hdf5 \
       --auto


Step 5: Run a small generation test
-----------------------------------

Run the migrated Franka cube stacking example in AutoData using the task and embodiment descriptors we created in Steps 2 and 3:

.. code-block:: bash

   python scripts/generate_dataset.py \
       --env_name Isaac-Stack-Cube-Franka-IK-Rel-v0 \
       --viz kit \
       --num_envs 10 \
       --alg mimicgen \
       --generation_num_trials 10 \
       --task_descriptor isaac_autodata_examples/tasks/franka_cube_stack.yaml \
       --embodiment isaac_autodata_examples/embodiments/franka_ik_rel.yaml \
       --input_file ./datasets/dataset_franka_annotated.hdf5 \
       --output_file ./datasets/generated_dataset_franka.hdf5

This command exercises the normal Isaac Lab environment, AutoData task descriptor, and AutoData
embodiment. Once it succeeds, the Franka migration
is complete. Increase ``--num_envs`` and ``--generation_num_trials`` for the full run. See the
:doc:`complete Franka cube-stacking workflow <franka_cube_stack_mimicgen/index>` for recording,
annotation, generation, and validation details.
