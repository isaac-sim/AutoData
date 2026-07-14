Embodiments
===========

An **embodiment descriptor** is a YAML file describing the robot from the data generator's
point of view: where to read its end-effector pose(s) in the observation buffer, and how to
convert between target end-effector poses and the environment's action vector. It is loaded
into an **embodiment adapter** — the object that performs those transforms at runtime — via
``embodiment_adapter_from_yaml()``.

This is the entire robot-specific surface of the framework: the generator itself never sees
joint names, controllers, or kinematics, only "give me the EEF pose" and "turn this target
pose into an action."

Adapter Types
-------------

The ``type:`` field selects the adapter class from ``EMBODIMENT_TYPE_REGISTRY``:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ``type``
     - Use for
   * - ``delta_pose_ik_single_arm``
     - Single arms driven by **relative** (delta) pose IK — e.g. the Franka tasks. The
       action is the clipped delta from the current EEF pose to the target, plus a gripper
       dimension.
   * - ``absolute_pose_whole_body_bimanual``
     - Humanoids driven by **absolute** pose targets through a whole-body IK controller —
       e.g. GR1T2 and G1. The action carries one absolute pose per arm plus hand-joint
       positions.

New morphology + controller combinations register a new adapter class in the registry.

What the Adapter Does at Runtime
--------------------------------

Every adapter is bound to the live environment once (``bind_env()``, done automatically when
the :doc:`Datastream <datastream>` is built) and then serves five queries. The first four are
the generator's entire view of the robot:

* ``get_eef_poses()`` — read each end-effector's current pose from the configured
  observation keys (``pose_obs_keys``).
* ``target_eef_pose_to_action()`` — the *forward* direction: turn per-EEF target poses plus
  the passthrough channels into one action for ``env.step()``, optionally adding action
  noise.
* ``action_to_target_eef_pose()`` — the *inverse* direction: recover the target poses encoded
  in a recorded action. This is how the source demos' controller targets are extracted.
* ``actions_to_passthrough_actions()`` — pull the non-pose channels (gripper or hand joints)
  out of recorded actions so the generator can replay them verbatim.

**Passthrough actions** are the action dimensions that carry no pose information — gripper
actuation, hand joints — and are copied from the source segment rather than recomputed.

The adapter also owns the robot's raw joint state (``get_joint_positions()``,
``get_joint_names()``), which motion planners read as their planning start state. Nothing
else in the framework touches the robot articulation directly.

Single-Arm Example (Franka, relative IK)
----------------------------------------

From :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/franka_ik_rel.yaml>`:

.. code-block:: yaml

   type: delta_pose_ik_single_arm
   name: franka_panda
   description: Franka 7-DOF arm with parallel gripper, delta-pose IK control.

   eef_name: franka

   pose_obs_keys:
     pos: eef_pos       # observation keys holding the EEF pose
     quat: eef_quat

   action_layout:
     gripper_dim: 1                   # trailing gripper dims in the action vector
     clip_pose_action_to_unit: true   # clip the delta-pose part to [-1, 1]

   eef_offset: [0.0, 0.0, 0.0]

The 7-D action is ``[delta position (3), delta rotation as axis-angle (3), gripper (1)]``.
The adapter computes the delta from the current EEF pose to the requested target, clips it,
and passes gripper actuation through from the source segment.

Bimanual Example (GR1, absolute IK)
-----------------------------------

From :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/gr1_ik_abs.yaml>`:

.. code-block:: yaml

   type: absolute_pose_whole_body_bimanual
   name: gr1t2

   eefs:
     left:
       pose_obs_keys: {pos: left_eef_pos, quat: left_eef_quat}
       gripper_action_indices: [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20]
     right:
       pose_obs_keys: {pos: right_eef_pos, quat: right_eef_quat}
       gripper_action_indices: [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 21]

   action_layout:
     left_pose_slice: [0, 7]     # pos (3) + quat (4)
     right_pose_slice: [7, 14]
     hand_joints_slice: [14, 36] # 11 DOF per hand
     canonicalize_quat: true

The 36-D action carries an absolute pose per arm plus 22 hand-joint positions; the adapter
assembles it from per-arm targets each step and reads each arm's pose back from its own
observation keys. ``gripper_action_indices`` map each arm's hand joints *within* the
hand-joints slice, so finger actuation replays per arm.

The ``eef_offset`` Frame Shift
------------------------------

``eef_offset`` is a translation (in meters) from the robot's kinematic control link to the
end-effector frame the adapter should report — equivalently, the difference between the
env-reported EEF frame and the frame the *source dataset's annotations* use. When the two
frames agree it is zero; when they do not, generation silently produces offset grasps, so
this is the first thing to check when transformed segments look shifted.

Concrete example: the base Franka IK-Rel tasks report the inter-fingertip ``end_effector``
frame, and the MimicGen source dataset is annotated in that same frame — offset zero. The
SkillGen cube-stack dataset, however, is annotated in the ``panda_hand`` frame, so its
embodiment config (``franka_ik_rel_skillgen.yaml``) sets ``eef_offset: [0, 0, 0.1034]`` —
the fingertip-to-hand distance.

Writing a New Embodiment
------------------------

1. **Identify the observations.** Find the observation keys holding each end-effector's
   position and quaternion (``pose_obs_keys``). The quaternion observation must be
   (w, x, y, z) ordered.
2. **Map the action vector.** Work out the pose slice(s), gripper dims/indices, and whether
   pose actions are relative or absolute — this picks the adapter ``type``.
3. **Determine ``eef_offset``.** Compare the env's reported EEF frame with the frame your
   source dataset's poses use.
4. **Write the YAML and round-trip it.** The adapter's two directions must be mutually
   consistent: converting an action to a target pose and back should reproduce the action.
   The unit tests in ``isaac_autodata_tests/interfaces/embodiments/`` show the pattern and
   run in seconds without Isaac Sim.

If no registered adapter fits the robot's control scheme, implement a new adapter class and
add it to ``EMBODIMENT_TYPE_REGISTRY``.
