Environment Profiles
====================

An **environment profile** is a YAML file that turns a base task into a variant — extra
objects in the scene, different reset randomization, a motion planner tuned for the new
clutter — without defining a new environment id. It is passed to the generation entry point
with ``--env_profile`` and overlaid onto the parsed env config just before the environment is
created.

The shipped example,
:autodata_code_link:`<autodata_examples/env_profiles/franka_bin_stack.yaml>`,
turns plain Franka cube stacking into stacking inside a narrow bin; the
:doc:`SkillGen workflow <../workflows/skillgen/index>` shows it in action.

Why Profiles Instead of New Environments?
-----------------------------------------

Task variants usually differ from their base task by a handful of scene and randomization
changes. Registering a new environment for each variant duplicates config code and multiplies
env ids; a profile keeps the base task as the single environment and expresses only the
delta:

* **One YAML per variant** — no new Python env config, no new registration.
* **Datasets keep working** — the base env id stays the same, so recorded and annotated
  source datasets can be reused across variants (SkillGen generates both cube-stacking tasks
  from one annotated dataset).
* **Reproducibility** — the generation-result JSON (``--result_file``) records the applied
  profile's name, path, and planner, since the output dataset itself only stores the base
  env id.

What a Profile Can Change
-------------------------

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Section
     - Effect
   * - ``scene.rigid_objects.add``
     - Spawn new rigid objects from USD files, with initial pose, scale, and rigid-body
       physics properties.
   * - ``scene.rigid_objects.override``
     - Patch the rigid-body properties of objects the base task already has (e.g. stiffer
       contact solving).
   * - ``events.remove``
     - Disable base-task event terms by name (e.g. drop the default cube randomization).
   * - ``events.override``
     - Merge new parameters into an existing event term.
   * - ``events.add``
     - Add new event terms — typically reset randomizations for the added or re-pinned
       objects.
   * - ``planner``
     - Name the motion-planner profile tuned for the modified scene (used by SkillGen; see
       :doc:`../advanced/motion_planners`).

Schema
------

.. code-block:: yaml

   name: <str>
   description: <str>                  # optional
   base_env: <env id>                  # must match the --env_name used at generation time
   planner: <planner profile name>     # optional; SkillGen only

   scene:                              # optional
     rigid_objects:
       add:
         <asset_name>:
           prim_path: "{ENV_REGEX_NS}/<Prim>"
           usd_path: "{ISAACLAB_NUCLEUS_DIR}/<...>.usd"
           position: [x, y, z]               # optional, env frame [m]
           rotation: [x, y, z, w]            # optional, identity is [0, 0, 0, 1]
           scale: [x, y, z]                  # optional
           rigid_props: {<field>: <value>}   # optional RigidBodyPropertiesCfg fields
       override:
         <asset_name>:
           rigid_props: {<field>: <value>}

   events:                             # optional
     remove: [<term_name>, ...]
     override:
       <term_name>: {params: {...}}
     add:
       <term_name>:
         func: "<module.path>:<function>"
         mode: reset                   # startup | reset | interval
         params: {...}

Profiles are validated on load; a schema violation raises an error naming the offending
field.

Field notes:

* ``usd_path`` may use the ``{ISAACLAB_NUCLEUS_DIR}`` and ``{ISAAC_NUCLEUS_DIR}`` placeholders
  to reference assets on the Nucleus server.
* ``rotation`` is an (x, y, z, w) quaternion — identity is ``[0, 0, 0, 1]``. A wrong order
  flips the asset by 180°.
* ``func`` references an event function as ``"module.path:function"``; it is resolved when
  the profile is applied.
* Inside event ``params``, entries named ``*_cfg`` / ``*_cfgs`` are treated as scene-asset
  names and converted to the entity references the event system expects.

How a Profile Is Applied
------------------------

``apply_env_profile()`` overlays the profile onto the parsed env config in place, before
``gym.make``. It is strict, and fails fast with a named reason rather than producing a
subtly wrong scene:

* ``base_env`` must equal the env id being created.
* ``add`` refuses to shadow an asset or event term the base task already defines;
  ``override`` and ``remove`` refuse to touch one it does not.
* Every scene asset referenced by event parameters must exist — on the base task or among
  the profile's additions — checked at apply time instead of failing later inside the env.

Changes are applied in order: scene additions, scene overrides, then events (remove →
override → add), so added events may reference added assets.

.. note::

   The spawn pose of an added object (``position`` / ``rotation``) is baked into the USD
   stage, and stage readers — such as the motion planner's collision world — see *that* pose,
   not the poses produced by reset events. If you pin an object with a reset event, give the
   spawn the same pose so planning and physics agree.

Writing a New Profile
---------------------

1. **Pick the base task** and set ``base_env`` to its env id.
2. **Describe the scene delta** — add new objects (USD path, spawn pose, scale) and override
   physics properties where the variant needs them.
3. **Adjust the reset events** — remove randomizations that no longer apply and add ones for
   the new layout; keep pinned objects' spawn poses in sync with their reset events.
4. **Pick or define a planner profile** if the variant is used with SkillGen, so the new
   objects become collision geometry (see :doc:`../advanced/motion_planners`).
5. **Verify with a short run** — generate a handful of attempts with ``--env_profile`` and
   ``--result_file``, and inspect the scene in a Kit window (``--viz kit``) before scaling up.
