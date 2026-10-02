# Cross-embodiment dataset retargeting

Replay a demonstration dataset recorded on one robot (the **source** embodiment) onto a different
robot (the **target** embodiment), one output demo per input demo. The source end-effector (EEF)
trajectory is transferred verbatim onto the target robot and the resulting rollout is recorded — no
object-centric regeneration happens, so the motion is copied 1:1 rather than re-planned.

Entry point: [`scripts/retarget_dataset.py`](../scripts/retarget_dataset.py). A run is driven by a
single **retarget descriptor** YAML (`--retarget_config`) that references two **embodiment
descriptor** YAMLs and the target env. Working examples for the Unitree **G1 ⇄** Fourier **GR1T2**
humanoid pair live in [`autodata_examples/`](../autodata_examples).

> All commands must run inside the project Docker container (`./docker/run_docker.sh`).

---

## 1. What it does

Given a source dataset and a source→target embodiment pair, retargeting:

1. reads each source episode's recorded **absolute EEF pose trajectory**
   (`obs/datagen_info/target_eef_pose`), which is embodiment-independent (env-relative SE(3));
2. resets the target env so the **task scene** (object poses) matches the source episode's initial
   state, while the **target robot** starts from its own default configuration;
3. re-encodes each step's EEF pose (plus gripper/hand commands) into the target embodiment's action
   and steps it through the target env's IK controller;
4. records the target rollout (actions + states, and optionally full `datagen_info`) as one output
   episode, marked success/failure by the target env's success term.

Because the trajectory is copied rather than regenerated, the source dataset **must already carry
`datagen_info`** — i.e. it was produced by `generate_dataset.py` or passed through
`annotate_demos.py`.

## 2. How it works

Per source episode, per EEF:

- **Reference trajectory.** The commanded/observed EEF path is extracted in a canonical *grasp
  frame*: `source_pose @ source_offset`, then re-anchored to the target's control link via
  `@ inv(target_offset)`. The per-EEF `offset` in each embodiment descriptor is the SE(3) transform
  from the tracked control link to that canonical grasp frame, so different wrists/grippers line up
  at the grasp point (handles gripper length and wrist convention). Which source signal drives the
  replay is set by `reference_pose` (observed `eef_pose` vs commanded `target_eef_pose`).

- **Segmentation into subtasks.** The descriptor's per-EEF `subtasks` list splits each demo into
  segments. Each subtask declares how it ends (`subtask_end`) from an event detected on the *source*
  demo — the gripper crossing open/closed, a subtask-term signal edge, or a fixed frame count.
  Segmentation drives object tracking, offsets, boundary settling, and (optionally) cross-arm
  synchronization.

- **Hand / gripper policy.** Gripper and other non-EEF channels (e.g. a locomotion `body` channel)
  are extracted from the source actions and remapped to the target's layout. `hand_policy` chooses
  how the source hand drives the target hand: `passthrough` (copy raw), `binary` (open/close by a
  closedness threshold), `interpolation` (scale between the target's `hand_open`/`hand_close`
  configs), or `joint_mapping` (wire target joints to source joints).

- **Object tracking.** When a subtask sets `object_tracking`, the EEF tracks the named object's pose
  (a rigid `eef_T_object`) over that segment instead of following the source EEF path — this keeps a
  carried object on its recorded world path even if the target's grasp differs slightly. The command
  eases between the EEF path and the object-centric pose over `interpolation_step_start` /
  `interpolation_step_after` steps.

- **Per-subtask offsets.** An optional `offset` on a subtask applies an SE(3) nudge to that EEF's
  commanded pose across the subtask's span, in a chosen frame (world / eef / the tracked object).
  Offsets form a continuous timeline: adjacent subtask offsets blend into each other, and the last
  subtask **holds** its offset (it is not decayed away), so an offset applied through the final
  placement stays applied through the success settle. Use it to correct a systematic grasp/place bias
  between the two embodiments.

- **Boundary settling.** At each subtask boundary the replayer holds the pose until the robot's
  joints (arm and gripper) stop moving, up to `segment_settle_steps` sim steps — motion-aware, so a
  fast gripper exits in ~1–2 steps while a slow linkage gets time to finish. A per-boundary
  `settle_steps` on `subtask_end` overrides the cap.

- **Success & recording.** After the trajectory completes, the pose is held for
  `success_settle_steps` while the target env's success term is re-checked. Episodes that satisfy it
  are exported; failures are dropped unless `--keep_failed` is set.

## 3. Configuration

A run is defined by two kinds of YAML file, both under
[`autodata_examples/`](../autodata_examples).

### 3.1 Embodiment descriptor (`autodata_examples/embodiments/*.yaml`)

Describes one robot's action layout so retargeting can read/write its EEF poses, hands, and
passthrough channels. Key sections (see [`g1_ik_abs.yaml`](../autodata_examples/embodiments/g1_ik_abs.yaml)
and [`gr1_ik_abs.yaml`](../autodata_examples/embodiments/gr1_ik_abs.yaml)):

| Field | Meaning |
| --- | --- |
| `type`, `name`, `description` | Adapter kind (e.g. `absolute_pose_whole_body_bimanual`) and labels. |
| `eefs.<eef>.pose_obs_keys` | Obs keys for this EEF's position / quaternion. |
| `eefs.<eef>.gripper_action_indices` | Which hand-joint action indices this EEF owns. |
| `eefs.<eef>.offset` | SE(3) transform (`axis_angle` [rad], `translation` [m]) from the tracked control link to the canonical grasp frame — how this robot's wrist aligns to the grasp point. |
| `action_layout` | Pose slices per EEF, hand-joint slice, `canonicalize_quat`, and any `passthrough_channels` (name → contiguous action slice, e.g. a `body` locomotion command). |
| `hand_open` / `hand_close` | Per-EEF fully-open / fully-closed hand joint configs, used to score source closedness and drive `binary` / `interpolation` hand policies. |

Both the **single-arm** (e.g. Franka) and **bimanual/whole-body** (G1, GR1) adapters are supported;
bimanual descriptors simply list two `eefs`.

### 3.2 Retarget descriptor (`autodata_examples/retarget/*.yaml`)

Bundles everything task/pair-specific into one file. Required references plus the most common knobs
(full schema in [`config.py`](config.py) — `RetargetConfig`):

| Field | Meaning |
| --- | --- |
| `source_embodiment` / `target_embodiment` | Paths (relative to this YAML) to the embodiment descriptors. |
| `target_env_name` | Gym env id to instantiate for the target robot. |
| `subtasks` | Per-EEF list of segments (see below). |
| `hand_policy` | `passthrough` / `binary` / `interpolation` / `joint_mapping`. |
| `reference_pose` | `eef_pose` (source *achieved* path, default) or `target_eef_pose` (commanded). |
| `eef_reference_link` | `controlled` rebuilds the reference at the IK-controlled link (needed when a robot observes the EEF a joint short of the link it drives, e.g. GR1). |
| `replay_speed` | Retime the source trajectory (>1 faster, <1 slower). |
| `retarget_frame`, `scene_translation` | Frame the trajectory is anchored in, and a world offset applied to the scene. |
| `target_channel_defaults` | Hold a target-only passthrough channel (e.g. G1's `body`) at a fixed value during replay. |
| `default_object_tracking` | Default `interpolation_step_start` / `interpolation_step_after` for object tracking. |
| `segment_settle_steps`, `success_settle_steps` | Boundary settle cap and post-success hold. |
| `write_datagen_info` | Write full `obs/datagen_info` so the output is a drop-in `generate_dataset.py` source. |
| `synchronization` | Barrier groups of subtask `name`s across EEFs that must conclude together (bimanual joins). |

**Subtasks.** Each EEF under `subtasks` is a list of segments in execution order. A segment has:

- `object_ref` / `frame_ref` — the object or frame it is planned relative to (also the default frame
  for its `offset`);
- `subtask_end` — how it ends: `method` is `gripper_open`/`gripper_close`
  (or `gripper_opening`/`gripper_closing` for the leading edge), `signal_on`/`signal_off`, or
  `fixed_length` (with `length`); `offset` shifts the boundary ± frames; `settle_steps` overrides the
  settle cap here. The **last** subtask of an EEF needs no `subtask_end` (it runs to the end);
- `object_tracking` — bare object name or a section, to track that object over the segment;
- `offset` — optional per-subtask SE(3) nudge (`frame`, `translation`, `axis_angle`,
  `interpolation_start`/`interpolation_end`);
- `name` — optional stable id (required only if referenced in `synchronization`).

Minimal example (right arm picks, then places, then returns):

```yaml
subtasks:
  right:
    - object_ref: world
      description: Approach and grasp the object with the right hand.
      subtask_end: { method: gripper_close, offset: 10 }
    - frame_ref: world
      description: Place the object in the bin.
      subtask_end: { method: gripper_open }
      object_tracking: object          # track the carried object over the transport
    - object_ref: world
      description: Return to idle.       # last subtask: no subtask_end
```

Adding a per-subtask offset (nudge the place 2 cm higher in world frame, eased in over 15 steps):

```yaml
    - frame_ref: world
      subtask_end: { method: gripper_open }
      object_tracking: object
      offset:
        frame: world
        translation: [0.0, 0.0, 0.02]
        interpolation_start: 15
```

## 4. Running

Config-based invocation (all task/pair parameters come from the descriptor):

```bash
python scripts/retarget_dataset.py \
    --retarget_config autodata_examples/retarget/g1_to_gr1_pick_place.yaml \
    --input_file  ./datasets/<source_g1_dataset>.hdf5 \
    --output_file ./datasets/retargeted_g1_to_gr1.hdf5 \
    --headless
```

The reverse direction just swaps the descriptor:

```bash
python scripts/retarget_dataset.py \
    --retarget_config autodata_examples/retarget/gr1_to_g1_pick_place.yaml \
    --input_file  ./datasets/<source_gr1_dataset>.hdf5 \
    --output_file ./datasets/retargeted_gr1_to_g1.hdf5 \
    --headless
```

### Runtime / IO flags (CLI only)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--retarget_config` | — | Retarget descriptor YAML (recommended path). |
| `--input_file` | *(required)* | Source dataset HDF5 (must carry `datagen_info`). |
| `--output_file` | `./datasets/retargeted_dataset.hdf5` | Destination HDF5. |
| `--num_envs` | `1` | `>1` replays episodes across that many parallel envs on one sim for a large speedup (the per-step tracking report is single-env only). |
| `--select_episodes IDX ...` | all | Retarget only these source-episode indices. |
| `--target_successes N` | — | Stop once `N` replays succeed. |
| `--target_runs N` | — | Stop once `N` replays are attempted. |
| `--keep_failed` | off | Also export episodes that did not satisfy the success term. |
| `--device`, `--headless`, … | — | Standard Isaac Lab `AppLauncher` flags. |

Without `--retarget_config`, only `--source_embodiment` / `--target_embodiment` /
`--target_env_name` are read and every other knob (including `subtasks`) takes its default, so a
descriptor is needed for any real run.

## 5. Output

The retargeted HDF5 records the target robot's actions and states. When the descriptor sets
`write_datagen_info: true`, the output also carries the full `obs/datagen_info` (observed `eef_pose`,
commanded `target_eef_pose`, per-object `object_pose`) and the forwarded `subtask_term_signals`,
making it a drop-in source for `generate_dataset.py` without a separate `annotate_demos.py` pass.
