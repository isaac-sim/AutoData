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

.. note::

   **Conventions used throughout.** Poses are 4×4 homogeneous matrices. Quaternions are
   **(x, y, z, w)** ordered — identity is ``[0, 0, 0, 1]`` — in observations, action layouts,
   and everywhere in between. Pose observations are read from the ``"policy"`` observation
   group unless ``obs_group`` says otherwise.

Adapter Types
-------------

The ``type:`` field selects the adapter class from ``EMBODIMENT_TYPE_REGISTRY``:

.. list-table::
   :widths: 34 44 22
   :header-rows: 1

   * - ``type``
     - Use for
     - Shipped examples
   * - ``delta_pose_ik_single_arm``
     - Single arms driven by **relative** (delta) pose IK — e.g. the Franka tasks. The
       action is the clipped delta from the current EEF pose to the target, plus a gripper
       dimension.
     - ``franka_ik_rel.yaml``, ``franka_ik_rel_skillgen.yaml``
   * - ``absolute_pose_whole_body_bimanual``
     - Humanoids driven by **absolute** pose targets through a whole-body IK controller.
       The action carries one absolute pose per arm plus hand-joint positions.
     - ``gr1_ik_abs.yaml`` (GR1T2), ``g1_ik_abs.yaml`` (G1)

New morphology + controller combinations register a new adapter class in the registry — see
`Writing a New Embodiment`_.

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
  in a recorded action. This is how the source demonstrations' controller targets are extracted.
* ``actions_to_passthrough_actions()`` — pull the non-pose channels (gripper or hand joints)
  out of recorded actions so the generator can replay them verbatim.

**Passthrough actions** are the action dimensions that carry no pose information — gripper
actuation, hand joints, a mobile-base command — and are copied from the source segment rather
than recomputed.

The adapter also owns the robot's raw joint state (``get_joint_positions()``,
``get_joint_names()``), which motion planners read as their planning start state. Nothing
else in the framework touches the robot articulation directly.

Single-Arm Adapter: ``delta_pose_ik_single_arm``
------------------------------------------------

The action vector is ``[delta_position (3), delta_rotation (3), gripper (gripper_dim)]`` —
``6 + gripper_dim`` values in total. The pose part is the delta from the current EEF pose to
the target; the rotation delta uses the compact axis-angle form (unit axis × angle in
radians).

From :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/franka_ik_rel.yaml>`:

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

Field reference
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 32 12 16 40
   :header-rows: 1

   * - Key
     - Required
     - Default
     - Meaning
   * - ``type``
     - yes
     - —
     - Adapter class; ``delta_pose_ik_single_arm`` here.
   * - ``name``
     - yes
     - —
     - Identifier for the embodiment (used in logs and registries).
   * - ``description``
     - no
     - ``""``
     - Human-readable summary.
   * - ``eef_name``
     - yes
     - —
     - Name of the single end-effector. Must match the EEF name used by the task
       descriptor's ``subtasks`` section.
   * - ``obs_group``
     - no
     - ``policy``
     - Observation-buffer group holding the pose keys.
   * - ``pose_obs_keys.pos``
     - yes
     - —
     - Observation key with the EEF position (3-vector [m]).
   * - ``pose_obs_keys.quat``
     - yes
     - —
     - Observation key with the EEF orientation quaternion, **(x, y, z, w)** ordered.
   * - ``action_layout.gripper_dim``
     - yes
     - —
     - Number of trailing action dimensions occupied by the gripper.
   * - ``action_layout.clip_pose_action_to_unit``
     - no
     - ``true``
     - Clamp the 6-D pose part of the action to ``[-1, 1]`` (after noise). Match this to
       what the env's action term expects.
   * - ``eef_offset``
     - no
     - ``[0, 0, 0]``
     - Translation [m] from the robot's kinematic control link to the frame the pose
       observation reports — see `The eef_offset Frame Shift`_.

The Transform, Step by Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is one 7-D Franka action followed through both directions. Suppose the current EEF pose
is at the origin with identity rotation, and the recorded action is::

   action = [0.10, 0.20, 0.30,   0.10, 0.00, 0.00,   0.7]
             └─ delta position ─┘└─ delta rotation ─┘ └ gripper

**Inverse — action → target pose** (``action_to_target_eef_pose``). Used when loading source
demonstrations: it recovers the controller *target* each recorded action encoded.

1. Split the action: ``delta_pos = [0.10, 0.20, 0.30]``, ``delta_aa = [0.10, 0, 0]``. The
   gripper value carries no pose and is ignored here.
2. Read the current EEF pose from the observation buffer (shifted by ``-eef_offset`` if one
   is configured).
3. Target position = current position + ``delta_pos`` → ``(0.10, 0.20, 0.30)``.
4. Target rotation = ``R(delta_aa) @ current_rotation`` — the axis-angle vector
   ``[0.10, 0, 0]`` means "rotate 0.10 rad about the x axis", composed onto the current
   orientation.

**Forward — target pose → action** (``target_eef_pose_to_action``). Used during generation:
each transformed waypoint becomes the action that is actually stepped.

1. Delta position = target position − current position.
2. Delta rotation = ``target_rotation @ current_rotationᵀ``, converted back to the compact
   axis-angle vector.
3. Optional **noise**: if the subtask sets ``action_noise``, Gaussian noise scaled by it is
   added to the 6-D pose part — never to the gripper.
4. Optional **clipping**: with ``clip_pose_action_to_unit: true``, the pose part is clamped
   to ``[-1, 1]`` — after the noise, so noise cannot push the action out of range.
5. The gripper value is appended **verbatim** from the source demonstration's passthrough
   channel — the adapter never invents gripper commands.

Running the forward direction on the inverse's output reproduces the original action (up to
noise and clipping). That round-trip property is exactly what the unit tests check — and what
you should check first for a new embodiment (see below).

Bimanual Adapter: ``absolute_pose_whole_body_bimanual``
-------------------------------------------------------

The action vector is
``[left_pos (3), left_quat (4), right_pos (3), right_quat (4), hand_joints, ...extras]``.
Poses are absolute targets tracked by a whole-body IK controller. The hand-joints block
interleaves both hands' joints in URDF order; ``gripper_action_indices`` records which
positions belong to which arm.

From :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/gr1_ik_abs.yaml>` (the
GR1T2 humanoid; :isaac_autodata_code_link:`<isaac_autodata_examples/embodiments/g1_ik_abs.yaml>`
has the same shape for the G1):

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
     hand_joints_slice: [14, 36] # 11 DOF per hand, interleaved
     canonicalize_quat: true

Field reference
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 32 12 16 40
   :header-rows: 1

   * - Key
     - Required
     - Default
     - Meaning
   * - ``type``
     - yes
     - —
     - Adapter class; ``absolute_pose_whole_body_bimanual`` here.
   * - ``name``
     - yes
     - —
     - Identifier for the embodiment.
   * - ``description``
     - no
     - ``""``
     - Human-readable summary.
   * - ``obs_group``
     - no
     - ``policy``
     - Observation-buffer group holding the pose keys.
   * - ``eefs.left`` / ``eefs.right``
     - yes
     - —
     - One block per arm. The EEF names are literally ``left`` and ``right``; the task
       descriptor's ``subtasks`` section must use the same names.
   * - ``eefs.<arm>.pose_obs_keys``
     - yes
     - —
     - ``pos`` / ``quat`` observation keys for that arm's EEF pose (quaternion
       (x, y, z, w)).
   * - ``eefs.<arm>.gripper_action_indices``
     - yes
     - —
     - That arm's positions *within* the hand-joints block (0-based, relative to the block,
       not the full action). Left and right must not overlap; the length is that arm's
       gripper action dim.
   * - ``action_layout.left_pose_slice``
     - yes
     - —
     - Half-open ``[start, end)`` range of the left ``[pos, quat]`` block. Must span
       exactly 7.
   * - ``action_layout.right_pose_slice``
     - yes
     - —
     - Range of the right pose block. Must immediately follow the left one.
   * - ``action_layout.hand_joints_slice``
     - yes
     - —
     - Range of the interleaved hand-joints block. Must immediately follow the right pose
       block.
   * - ``action_layout.canonicalize_quat``
     - no
     - ``true``
     - Flip quaternions to ``w >= 0`` before packing into the action (``q`` and ``-q`` are
       the same rotation; canonicalizing prevents sign flips between consecutive recorded
       actions).
   * - ``action_layout.passthrough_channels``
     - no
     - ``{}``
     - Extra **non-EEF** passthrough channels, ``name: [start, end)`` each — e.g. a
       mobile-base or locomotion command. Copied verbatim from the source demonstration like the
       grippers. Slices must not overlap the pose + hand-joints region, and names must not
       collide with ``left`` / ``right``.

All slice layouts are validated at load time — spans, adjacency, index ranges, and overlaps
fail immediately with a named reason.

The ``eef_offset`` Frame Shift
------------------------------

``eef_offset`` (single-arm only) is a translation [m] from the robot's kinematic control link
to the end-effector frame the observation reports — equivalently, the difference between the
env-reported EEF frame and the frame the *source dataset's annotations* use. When the two
frames agree it is zero; when they do not, generation silently produces offset grasps, so
this is the first thing to check when transformed segments look shifted.

Concrete example: the base Franka IK-Rel tasks report the inter-fingertip ``end_effector``
frame, and the MimicGen source dataset is annotated in that same frame — offset zero. The
SkillGen cube-stacking dataset, however, is annotated in the ``panda_hand`` frame (cuRobo's
planning link), so its embodiment config (``franka_ik_rel_skillgen.yaml``) sets
``eef_offset: [0, 0, 0.1034]`` — the fingertip-to-hand distance.

Writing a New Embodiment
------------------------

A new embodiment for an already-supported control scheme is pure YAML — no code. Work
through these steps:

1. **Identify the observations.** Find the observation keys holding each end-effector's
   position and quaternion (``pose_obs_keys``), and which group they live in
   (``obs_group``). The quaternion observation must be (x, y, z, w) ordered.
2. **Map the action vector.** Work out the pose slice(s), gripper dims/indices, and whether
   pose actions are relative or absolute — this picks the adapter ``type``. Anything that is
   neither pose nor gripper becomes a ``passthrough_channels`` entry.
3. **Determine ``eef_offset``.** Compare the env's reported EEF frame with the frame your
   source dataset's poses use (single-arm; see above).
4. **Round-trip it in a unit test.** The adapter's two directions must be mutually
   consistent: converting an action to a target pose and back must reproduce the action.
   This runs in seconds, without Isaac Sim:

   .. code-block:: python

      import torch

      from isaac_autodata_interfaces.embodiments import embodiment_adapter_from_yaml
      from isaac_autodata_tests.interfaces.mocks import MockEnv


      def test_my_embodiment_round_trip():
          adapter = embodiment_adapter_from_yaml("my_robot.yaml")
          # Fake the env: only the pose observations the adapter reads.
          adapter.bind_env(
              MockEnv(
                  obs_buf={
                      "policy": {
                          "eef_pos": torch.zeros(1, 3),
                          "eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),  # (x, y, z, w)
                      }
                  }
              )
          )

          action_in = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.7]])
          target = adapter.action_to_target_eef_pose(action_in)["my_eef"][0]
          action_out = adapter.target_eef_pose_to_action(
              {"my_eef": target}, {"my_eef": action_in[0, 6:]}, env_id=0
          )

          assert torch.allclose(action_out, action_in[0], atol=1e-5)

   The shipped tests in ``isaac_autodata_tests/interfaces/embodiments/`` show the full
   pattern (disable clipping for large test deltas, batch shapes, per-arm variants). Run
   them with ``pytest isaac_autodata_tests/interfaces/embodiments/``.
5. **Verify against real data.** Replay a recorded demonstration through
   ``action_to_target_eef_pose`` and check the recovered targets track the recorded EEF
   poses.

If no registered adapter fits the robot's control scheme, implement a new adapter class and
add it to ``EMBODIMENT_TYPE_REGISTRY`` — the two existing adapters are the template: subclass
the morphology base class, implement the three action-encoding methods, and provide
``from_dict``.

Troubleshooting
---------------

.. list-table::
   :widths: 36 32 32
   :header-rows: 1

   * - Symptom
     - Likely cause
     - Fix
   * - Every grasp lands offset from the object by the same fixed distance.
     - ``eef_offset`` does not match the frame difference between the env's EEF observation
       and the source dataset's annotation frame.
     - Measure the offset between the two frames and set ``eef_offset`` accordingly — see
       `The eef_offset Frame Shift`_.
   * - Assets or motions come out flipped ~180°, or rotations drift wildly.
     - Quaternion order mismatch: a (w, x, y, z) quaternion fed where (x, y, z, w) is
       expected, or vice versa.
     - The framework is (x, y, z, w) throughout, identity ``[0, 0, 0, 1]``. Check the env's
       observation term and any hand-written poses.
   * - Large motions execute slowly or fall short; fast recorded motions replay truncated.
     - The 6-D pose action saturates at the ``[-1, 1]`` clamp
       (``clip_pose_action_to_unit``).
     - Expected for delta-IK envs whose action term clips — per-step deltas should be
       small. If the env does not clip, set ``clip_pose_action_to_unit: false``.
   * - ``KeyError`` on an observation key at the first pose read.
     - ``pose_obs_keys`` / ``obs_group`` don't match the env's observation terms.
     - List the env's observation terms and copy the exact key and group names.
   * - Fingers don't move — or the wrong hand's fingers move (bimanual).
     - ``gripper_action_indices`` misassigned between the arms, or indexed relative to the
       full action instead of the hand-joints block.
     - Indices are 0-based *within* ``hand_joints_slice``; compare against the env's URDF
       joint order.
   * - ``AssertionError: env already bound`` or ``Call bind_env(env) ...``.
     - The adapter was bound twice, or used before binding.
     - Build the adapter and hand it to the :doc:`Datastream <datastream>` — it binds the
       env exactly once.
