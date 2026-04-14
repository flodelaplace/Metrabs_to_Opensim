# MeTRAbs to OpenSim

**From monocular video to biomechanical analysis in one command.**

This pipeline extracts 3D human poses from a single-camera video using [MeTRAbs](https://github.com/isarandi/metrabs) and produces OpenSim-compatible outputs (TRC marker files, scaled models, inverse kinematics).

```bash
python run.py demo/cam01.mp4 --ik --height 1.80 --mass 75
```

---

## Why this approach?

### Classical pipeline (2 steps)

Most markerless motion capture pipelines work in two stages:

1. **2D pose estimation** (e.g. OpenPose, MediaPipe, MMPose) — detects joint positions in the image plane
2. **3D lifting** — a separate model reconstructs 3D coordinates from the 2D detections

This introduces error at each stage, requires careful calibration between steps, and typically produces **17 to 33 keypoints** (COCO, H36M conventions).

### This pipeline (1 step)

MeTRAbs (**Me**tric-Scale **Tr**uncation-Robust **Ab**solute Heatmap**s**) estimates **absolute 3D poses directly** from each camera image in a single forward pass. No intermediate 2D stage, no separate lifting model.

Key advantages:

| | Classical (2D + lifting) | MeTRAbs (direct 3D) |
|---|---|---|
| **Steps** | 2 (detection + lifting) | 1 (end-to-end) |
| **Keypoints** | 17-33 (COCO, H36M) | **87** (`bml_movi_87`) |
| **Output** | Relative 3D (root-centered) | **Absolute 3D** (metric-scale, in mm) |
| **Truncation** | Fails on partial views | Robust to cropping |
| **Anatomical detail** | Major joints only | Joints + anatomical landmarks (ASIS, PSIS, scapula, C7, etc.) |

The 87-joint `bml_movi_87` skeleton includes anatomical landmarks needed for biomechanical analysis that are absent from standard pose estimation skeletons, making it directly usable with OpenSim musculoskeletal models.

---

## Features

| Feature | Description |
|---|---|
| **Direct 3D pose** | MeTRAbs EfficientNetV2-L model, `bml_movi_87` skeleton (87 joints) |
| **TRC export** | OpenSim-compatible marker file (mm, Y-up) |
| **JSON export** | Full data: poses3d, poses2d, detection confidence |
| **OpenSim IK** | Automatic scaling + Inverse Kinematics (`--ik`) |
| **Multi-person** | ByteTrack tracking with per-person output (`--multi_person`) |
| **Per-person anthropometry** | Height/mass per person (`--person_heights`, `--person_masses`) |
| **Combined TRC** | All persons in one TRC for visualization (`--combined_trc`) |
| **Butterworth filter** | 6 Hz low-pass, zero-phase, 4th order (biomechanics standard) |
| **Auto-straighten** | Vertical calibration from standing frames |
| **Height rescaling** | Scale to real subject height from standing frames |
| **Flip correction** | Fixes lateral and anterior-posterior depth ambiguity |
| **Stationary mode** | Centers pelvis horizontally, detects flight phases (`--stationary`) |

---

## Installation

### Platform support

| Platform | Status |
|---|---|
| **Linux** (Ubuntu 22.04) | Tested, fully supported |
| **Windows via WSL2** | Tested, fully supported (recommended for Windows users) |
| **Windows native** | Not tested — CUDA/TensorFlow GPU setup differs significantly (system-wide NVIDIA CUDA install required instead of conda). CPU-only mode may work. If you're on Windows, **use WSL2**. |
| **macOS** | Not supported (requires NVIDIA GPU) |

### Prerequisites

- Linux or WSL2 (tested on Ubuntu 22.04)
- NVIDIA GPU with recent drivers (tested with RTX 3500 Ada)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

### 1. Clone the repository

```bash
git clone https://github.com/flodelaplace/Metrabs_to_Opensim.git
cd Metrabs_to_Opensim
```

### 2. Create the conda environment

```bash
conda env create --file environment.yml
conda activate metrabs_opensim
```

### 3. Configure GPU (CUDA)

TensorFlow installed via pip needs the CUDA library path set manually:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' \
  > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Reactivate to apply
conda deactivate && conda activate metrabs_opensim
```

Verify GPU access:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 4. Install OpenSim (optional, for `--ik`)

```bash
conda install -c opensim-org opensim
```

Verify:

```bash
python -c "import opensim; print(opensim.GetVersion())"
```

### 5. MeTRAbs model (automatic)

The MeTRAbs model is downloaded automatically on first run via TensorFlow Hub (~700 MB). It is cached in `~/.cache/tfhub_modules/` — subsequent runs start instantly.

No manual download is needed.

### Tested versions

| Component | Version |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.12.0 |
| CUDA Toolkit | 11.8 |
| cuDNN | 8.9 |
| OpenSim | 4.5+ |
| supervision | 0.27+ |
| NVIDIA Driver | 581+ |

---

## Quick start

Run the included demo video:

```bash
conda activate metrabs_opensim
python run.py demo/cam01.mp4
```

This will:
1. Download the MeTRAbs model (first run only)
2. Run 3D pose estimation on every frame
3. Apply filtering, reorientation, and ground calibration
4. Save results to `output/cam01_<timestamp>/`

With OpenSim IK:

```bash
python run.py demo/cam01.mp4 --ik --height 1.80 --mass 75
```

---

## Usage

### Single person (default)

```bash
python run.py video.mp4
python run.py video.mp4 --ik --height 1.80 --mass 75
python run.py video.mp4 --stationary --ik --height 1.80 --mass 75
```

### Multi-person

```bash
python run.py video.mp4 --multi_person
python run.py video.mp4 --multi_person --ik --height 1.75 --mass 70
python run.py video.mp4 --multi_person --ik --combined_trc \
    --person_heights 1.62,1.70,1.78 --person_masses 55,70,80
```

### All parameters

| Parameter | Default | Description |
|---|---|---|
| `video` | *(required)* | Path to video file or URL |
| `--output_dir` | `output/` | Base output directory |
| `--ik` | off | Run OpenSim scaling + Inverse Kinematics |
| `--multi_person` | off | Track all persons separately (ByteTrack) |
| `--stationary` | off | Center pelvis horizontally (squat, CMJ, treadmill) |
| `--combined_trc` | off | Multi-person: single combined TRC file |
| `--min_track_seconds` | `2.0` | Minimum track duration for multi-person (seconds) |
| `--height` | `1.75` | Subject height in meters |
| `--mass` | `69` | Subject mass in kg |
| `--person_heights` | - | Comma-separated heights per person, left-to-right |
| `--person_masses` | - | Comma-separated masses per person, left-to-right |

### Per-person anthropometry (multi-person)

In `--multi_person` mode, persons are sorted **left-to-right** by their pelvis X position on the first detected frame. The `--person_heights` and `--person_masses` lists follow this order:

```bash
python run.py video.mp4 --multi_person --ik --combined_trc \
    --person_heights 1.62,1.64,1.61,1.70,1.78,1.75 \
    --person_masses 55,60,58,70,80,75
```

If the list is shorter than the number of detected persons, extras fall back to `--height` / `--mass`.

---

## Output

### Single person

```
output/<video_name>_<YYYYMMDD_HHMMSS>/
    poses3d.trc              # TRC markers for OpenSim (mm, Y-up)
    results.json             # Full data (poses3d, poses2d, confidence)
    poses3d_scaled.osim      # Scaled OpenSim model       (if --ik)
    poses3d_ik.mot           # Joint angles from IK        (if --ik)
```

### Multi-person (`--multi_person`)

```
output/<video_name>_<YYYYMMDD_HHMMSS>/
    summary.json                 # Track info per person
    poses3d_combined.trc         # All persons combined    (if --combined_trc)
    person_0/                    # Leftmost person
        poses3d.trc
        results.json
        poses3d_scaled.osim      (if --ik)
        poses3d_ik.mot           (if --ik)
    person_1/
        ...
```

### JSON structure

```json
{
  "fps": 30.0,
  "joint_names": ["backneck", "upperback", "..."],
  "start_frame": 0,
  "frames": [
    {
      "frame": 1,
      "time": 0.0,
      "num_persons": 1,
      "persons": [
        {
          "confidence": 0.9532,
          "poses3d": [[x, y, z], ...],
          "poses2d": [[x, y], ...]
        }
      ]
    }
  ]
}
```

> **Note:** Confidence is per detection (person-level), not per joint.

---

## Visualizing results in OpenSim GUI

After running the pipeline with `--ik`, you get a scaled model (`.osim`) and joint angles (`.mot`). To visualize the motion in OpenSim GUI:

### 1. Open the scaled model

- **File > Open Model...**
- Navigate to `output/<video_name>_<timestamp>/` (or `person_0/` in multi-person mode)
- Select `poses3d_scaled.osim`
- The model appears in the 3D viewport, scaled to the subject's anthropometry

### 2. Load the motion

- **File > Load Motion...**
- Select `poses3d_ik.mot` (in the same folder as the model)
- The motion loads into the **Navigator** panel on the left

### 3. Play the motion

- Use the **slider** at the bottom of the viewport to scrub through the motion
- Click the **play button** to animate
- Right-click the motion in the Navigator to adjust playback speed

### 4. Inspect joint angles

- **Tools > Plot** to open the Plotter
- Select joints of interest (e.g. `knee_angle_r`, `hip_flexion_l`) to visualize the kinematics curves over time

### Tips

- If the model appears underground or floating, the ground calibration may need adjustment — check the TRC visually first
- To visualize markers on the model: **View > Markers** (toggles marker visibility)
- To compare with the raw TRC markers: **File > Load Motion...** and select `poses3d.trc` — this shows the marker trajectories overlaid on the model

---

## Processing pipeline

```
Video frames
    |
    v
[MeTRAbs inference] -------> raw 3D poses (camera frame, mm)
    |
    v
[Person selection] ---------> largest bbox (single) or ByteTrack (multi)
    |
    v
[Butterworth filter] -------> 6 Hz low-pass, 4th order, zero-phase
    |
    v
[Axis remap] ---------------> camera (X-right, Y-down, Z-forward)
    |                          to OpenSim (X-anterior, Y-up, Z-right)
    v
[Flip correction] ----------> fixes lateral + anterior-posterior ambiguity
    |
    v
[Standing detection] -------> knee angle > 160 deg on both legs
    |
    v
[Vertical straighten] ------> ankle-pelvis aligned with Y (from standing frames)
    |
    v
[Height rescaling] ---------> head-to-heel scaled to --height
    |
    v
[Ground calibration] -------> feet at Y=0 (5th percentile)
    |
    v
[Stationary centering] -----> pelvis centered on X,Z (if --stationary)
    |
    v
[TRC + JSON export]
    |
    v
[OpenSim Scaling + IK] -----> .osim + .mot (if --ik)
```

### Coordinate system

| Axis | OpenSim convention | Camera convention |
|---|---|---|
| X | Anterior (forward) | Image right |
| Y | Superior (up) | Image down |
| Z | Person's right (lateral) | Depth (forward) |

The axis remap assumes the **subject faces the camera**:
`X_osim = -Z_cam`, `Y_osim = -Y_cam`, `Z_osim = -X_cam`.

---

## OpenSim integration

### Marker set

The `bml_movi_87` marker set maps all 87 joints to OpenSim body segments:

- **Anatomical markers** (55): positions from the SKEL model (lab-grade surface landmarks)
- **Joint centers** (19): positions from HALPE_26 and LSTM models
- **Custom** (13): estimated positions for markers without direct equivalents

### Model

Uses `Model_Pose2Sim_muscles_flex.osim` — a musculoskeletal model with 62 DOF and 318 muscles, from the [Pose2Sim](https://github.com/perfanalytics/pose2sim) project.

### Setup files

| File | Purpose |
|---|---|
| `Markers_bml_movi_87.xml` | 87 markers with body attachments and positions |
| `Scaling_Setup_bml_movi_87.xml` | 11 measurement pairs for segment scaling |
| `IK_Setup_bml_movi_87.xml` | 87 IK tasks with weights |
| `Model_Pose2Sim_muscles_flex.osim` | Unscaled musculoskeletal model |
| `Geometry/` | 215 mesh files for model visualization |

### IK weights

| Body region | Weight | Rationale |
|---|---|---|
| Pelvis (ASIS, PSIS, hip centers) | 25 | Anchors the model |
| Knees, ankles | 30 | Critical for gait |
| Heels | 60 | Ground contact reference |
| Shoulders, elbows, wrists | 5 | Upper body articulation |
| Thigh, shin, forearm | 4 | Segment tracking |
| Torso (C7, scapula, back) | 2-5 | Trunk orientation |
| Hands (fingers, thumb) | 1 | Low priority |
| Head | 0.5 | Low reliability from video |

---

## Multi-person tracking

When `--multi_person` is used:

1. **Detection**: MeTRAbs detects all persons per frame
2. **Tracking**: [ByteTrack](https://github.com/ifzhang/ByteTrack) (via [supervision](https://github.com/roboflow/supervision)) assigns stable IDs
3. **Filtering**: tracks shorter than `--min_track_seconds` are discarded
4. **Interpolation**: missing frames within a track are linearly interpolated
5. **Export**: each person gets independent post-processing and their own subfolder

---

## Known limitations

| Limitation | Description | Workaround |
|---|---|---|
| **Monocular depth** | Depth estimated from apparent size, can drift | Use `--stationary` for fixed-camera setups |
| **Jump height** | Vertical displacement can be exaggerated | Joint angles (IK) are still valid |
| **Depth ambiguity** | Occasional front/back flip | Auto-corrected by flip detection |
| **Facing direction** | Assumes subject faces camera | Adjust code for other orientations |
| **Single camera** | No true 3D triangulation | Use [Pose2Sim](https://github.com/perfanalytics/pose2sim) for multi-camera |

---

## Project structure

```
Metrabs_to_Opensim/
    run.py                  # Main pipeline (inference + post-processing + export)
    kinematics.py           # OpenSim scaling + Inverse Kinematics
    environment.yml         # Conda environment
    opensim_setup/
        Markers_bml_movi_87.xml
        Scaling_Setup_bml_movi_87.xml
        IK_Setup_bml_movi_87.xml
        Model_Pose2Sim_muscles_flex.osim
        Geometry/           # 215 mesh files
    demo/
        cam01.mp4           # Demo video
    tools/
        calibrate_markers.py  # Advanced: recalibrate marker positions
    output/                 # Generated at runtime (gitignored)
```

---

## Performance

Benchmarked on RTX 3500 Ada (12 GB VRAM):

| Step | Time |
|---|---|
| MeTRAbs inference (GPU) | ~0.8 s/batch (8 frames) |
| MeTRAbs inference (CPU) | ~11 s/batch |
| Butterworth filter | < 0.1s / 1000 frames |
| OpenSim Scaling | ~2s |
| OpenSim IK | ~5s / 300 frames |

---

## Credits

This project builds on:

- **[MeTRAbs](https://github.com/isarandi/metrabs)** by Istvan Sarandi et al. — 3D pose estimation model
  - [MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://arxiv.org/abs/2007.07227) (T-BIOM 2021)
  - [Learning 3D Human Pose Estimation from Dozens of Datasets](https://arxiv.org/abs/2212.14474) (WACV 2023)
- **[Pose2Sim](https://github.com/perfanalytics/pose2sim)** by David Pagnon et al. — OpenSim model and marker set design
- **[supervision](https://github.com/roboflow/supervision)** by Roboflow — multi-person tracking (ByteTrack)

---

## License

MIT License. See [LICENSE](LICENSE).

> **Note:** The MeTRAbs pre-trained models can only be used for **non-commercial purposes** due to the licensing of the training datasets.
