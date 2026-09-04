Generation Algorithms
=====================

AutoData ships three generation algorithms, selected with ``--alg`` on the CLI (the
task descriptor's ``algo`` field independently selects which per-subtask ``algo_params``
schema the descriptor is parsed with — keep the two consistent). All three plug into the
:doc:`data generator <data_generator>` and share the same skeleton — split
source demonstrations into subtask segments, select a segment per subtask, transform it to the
current scene, execute, and keep successful demonstrations — and differ in how segments are bridged and how
multiple arms are handled.

.. list-table::
   :widths: 18 14 18 50
   :header-rows: 1

   * - Algorithm
     - Arms
     - Bridging
     - Notes
   * - ``mimicgen``
     - 1
     - Interpolation
     - Vanilla single-arm MimicGen.
   * - ``dexmimicgen``
     - 2
     - Interpolation
     - Adds per-arm subtasks and cross-arm coordination constraints.
   * - ``skillgen``
     - 1
     - Motion planning
     - cuRobo-planned, collision-free transit; requires start signals.

MimicGen
--------

MimicGen (`Mandlekar et al., 2023 <https://arxiv.org/abs/2310.17596>`_) is the foundation:
every subtask segment is stored relative to its reference object, so replaying it under a new
object pose is a rigid-body transform of the recorded end-effector trajectory. Between
segments, the end-effector linearly interpolates from its current pose to the transformed
segment start (``num_interpolation_steps``, optionally holding ``num_fixed_steps``), and
per-step ``action_noise`` adds diversity across attempts. An attempt succeeds if the task's
success condition holds at any step; only then is the demonstration exported (subject to the
:doc:`generation policy <task_descriptors>`).

The transformation requires **task-space actions**: the embodiment's actions must encode
end-effector pose targets (relative or absolute), which is what the
:doc:`embodiment adapters <embodiments>` guarantee.

DexMimicGen
-----------

DexMimicGen (`Jiang et al., 2024 <https://arxiv.org/abs/2410.24185>`_) extends the recipe to
two arms. Each end-effector declares its own subtask sequence in the descriptor, and segments
are selected, transformed, and executed per arm. Two mechanisms keep the arms consistent:

* **Source selection scope** — by default all arms replay segments from the same source demonstration
  (``select_src_per_arm: false``), preserving whatever implicit coordination the human
  demonstration had.
* **Constraints** — explicit ``sequential`` / ``coordination`` entries in the descriptor
  synchronize specific subtask pairs across arms at runtime (see
  :doc:`task_descriptors`).

SkillGen
--------

SkillGen (`Garrett et al., 2024 <https://arxiv.org/abs/2410.18907>`_) replaces interpolation
with **motion-planned transit**: each subtask executes as a collision-free cuRobo plan to the
skill segment's start, followed by the transformed skill replay. Grasped objects are attached
to the kinematic chain during planning, so held objects are collision-checked too. This
requires knowing exactly where skills start — hence subtask **start signals** and a
termination signal on every subtask, including the final one. Constraints apply only during
skill segments; transit is constraint-free. If planning fails for an attempt, the attempt is
abandoned and counted as a failure.

See the :doc:`SkillGen workflow <../workflows/skillgen/index>` and
:doc:`../advanced/motion_planners`.

Source Demonstration Selection Strategies
-----------------------------------------

Which source segment is replayed for a subtask is decided by a **selection strategy**,
configured per subtask in the task descriptor (``selection_strategy`` /
``selection_strategy_kwargs``):

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Strategy
     - Behavior
   * - ``random``
     - Uniform random choice over source demonstrations.
   * - ``nearest_neighbor_object``
     - Ranks demonstrations by how close their reference-object pose is to the current one, then
       picks randomly among the ``nn_k`` nearest.
   * - ``nearest_neighbor_robot_distance``
     - Ranks demonstrations by how close their (object-frame-transformed) source end-effector pose
       lands to the current end-effector pose, then picks among the ``nn_k`` nearest.

The nearest-neighbor strategies trade diversity for reliability: segments recorded under
similar geometry transform with less distortion, raising the success rate, while ``nn_k``
keeps some randomness so the output does not collapse onto a single source demonstration. The shipped
examples use ``nearest_neighbor_object`` with ``nn_k: 3``. How often selection re-runs — per
subtask, per arm, or once per episode — is set by the generation policy's
``select_src_per_subtask`` / ``select_src_per_arm`` flags.
