..
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Compatibility Matrix
====================

This matrix shows which robots, objects, generation algorithms, and physics engines work together in
this version of AutoData. Combinations that are not listed are not validated. For hardware, operating
system, and the pinned Isaac Sim, Isaac Lab, and Isaac Lab-Arena versions, see the :doc:`support_matrix`.

.. list-table:: AutoData generation compatibility
   :widths: auto
   :header-rows: 1
   :stub-columns: 1

   * - Robot
     - Actions [#actions]_
     - Algorithm
     - Motion planning
     - Objects
     - PhysX
     - Newton
     - Example
   * - Single arm
     - Relative pose
     - MimicGen
     - No
     - Rigid
     - ✓
     - —
     - :doc:`Franka cube stacking <../workflows/franka_cube_stack_mimicgen/index>`
   * -
     - Relative pose
     - SkillGen
     - Yes [#planning]_
     - Rigid
     - ✓
     - —
     - :doc:`Franka cube and bin stacking <../workflows/skillgen/index>`
   * -
     - Relative pose
     - SoftMimicGen
     - No
     - Deformable [#deformable]_
     - ✓
     - ✓ [#newton]_
     - :doc:`Franka rope <../workflows/franka_rope_softmimicgen/index>`
   * - Two arms
     - Absolute pose
     - DexMimicGen
     - No
     - Rigid
     - ✓
     - —
     - :doc:`Fourier GR-1 and Unitree G1 pick-and-place <../workflows/humanoid_dexmimicgen/index>`

✓ supported, with a shipped example · — not validated

.. [#actions] The example's embodiment adapter sends end-effector pose actions. A robot that takes other actions,
   such as joint positions, needs a new :doc:`embodiment adapter <../concepts/embodiments>` that converts between
   pose targets and those actions.
.. [#planning] SkillGen plans collision-aware transit between skills with cuRobo, an
   :doc:`optional installation <installation>`. Planner settings ship for the Franka; other robots need their
   own.
.. [#deformable] Record, annotate, generate, and replay with the same deformable asset and physics engine.
.. [#newton] Newton simulates the robot with MuJoCo Warp and the deformable object with VBD (Vertex Block
   Descent), and couples the two solvers.

AutoData runs in Isaac Lab manager-based environments, including Isaac Lab-Arena environments. Isaac Lab
direct-workflow environments are not supported.
