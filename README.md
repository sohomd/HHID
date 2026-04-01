# HHID: A Human-Human Interaction Dataset for Contact-Aware Novel View Synthesis

# Preview Dataset Samples
Synchronised Camera Capture

<table>
  <tr>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/HHID/assets/CAM1.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/HHID/assets/CAM2.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/HHID/assets/CAM3.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/HHID/assets/CAM4.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>

  
## Requirements
linux=22.04.5; Windows 10 or above; python>= 3.8;

## Dataset
<a href="https://tinyurl.com/2t92ktrw" target="_blank" rel="noopener noreferrer">Download Here</a>

### Evaluation Protocol

- **Rendering:** NVS methods generate outputs from the full scene (union of all subjects).
- **Metrics:** Occlusion and ISR are computed using identity-aware instance masks (A/B).

Separation ensures accurate measurement of interaction and identity consistency.

This repository provides evaluation scripts for:
- **Occlusion (IoU-based contact measure)**
- **Identity Switching Rate (ISR)**

## Installation
Configure the environment
```setup
conda env create --file environment.yml
```
## Occlusion metrics
Run the below script in the Data folder (change the path in main function if needed)

Per-frame per-camera occlusion
```setup
python calculate_occlusion.py
```
Average occlusion on all the camera
```setup
python avg_occlusion.py
```
## 📂 Expected Folder Structure

```bash
Data
├── Camera1/
│   ├── Instance Mask/
│   │   └── A/
│   │   └── B/
│   │       
│   └── PRED/
│       └── Instance Mask/
│           ├── A/
│           └── B/
│
├── Camera2/
├── Camera3/
└── Camera4/
Testdata/
```
## Evaluation Metric (ISR):
Calculate Identity Switching Rate (change file path in main function if needed)
```setup
python calculate_ISR.py
```
