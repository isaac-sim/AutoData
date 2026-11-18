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
     - Algorithm
     - Motion planning
     - Objects
     - PhysX
     - Newton
     - Example
   * - Single arm
     - MimicGen
     - No
     - Rigid
     - ✓
     - —
     - :doc:`Franka cube stacking <../workflows/franka_cube_stack_mimicgen/index>`
   * -
     - SkillGen
     - Yes [#planning]_
     - Rigid
     - ✓
     - —
     - :doc:`Franka cube and bin stacking <../workflows/skillgen/index>`
   * -
     - SoftMimicGen
     - No
     - Deformable [#deformable]_
     - ✓
     - ✓ [#newton]_
     - :doc:`Franka rope <../workflows/franka_rope_softmimicgen/index>`
   * - Two arms
     - DexMimicGen
     - No
     - Rigid
     - ✓
     - —
     - :doc:`Fourier GR-1 and Unitree G1 pick-and-place <../workflows/humanoid_dexmimicgen/index>`

✓ supported, with a shipped example · — not validated

.. [#planning] SkillGen plans collision-aware transit between skills with cuRobo, an
   :doc:`optional installation <installation>`. Planner settings ship for the Franka; other robots need their
   own.
.. [#deformable] Record, annotate, generate, and replay with the same deformable asset and physics engine.
.. [#newton] Newton simulates the robot with MuJoCo Warp and the deformable object with VBD (Vertex Block
   Descent), and couples the two solvers.

* **Environments:** AutoData runs in Isaac Lab manager-based environments, including Isaac Lab-Arena
  environments. Isaac Lab direct-workflow environments are not supported.
* **Robot control:** The environment must accept end-effector pose actions, relative or absolute. A robot
  controlled only in joint space needs a new :doc:`embodiment adapter <../concepts/embodiments>`.
