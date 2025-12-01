# CellCounter Biomarker Detection Pipeline

CellCounter is a computer-vision + deep-learning pipeline for detecting microscopic biomarkers in tissue-stained microscopy images.  
It uses color-space gating, candidate generation, CNN classification, geometric postprocessing, and exports DDG-compatible `.pnt` annotation files for use inside DotDotGoose.

---

## Purpose

This tool automates the process of detecting biomarkers that would otherwise be manually counted by a human using DotDotGoose.

This enables:

- automated microscopy detection
- reproducible & consistent annotation
- rapid dataset labeling
- direct interoperability with DotDotGoose
- hybrid AI + human review workflows

---

## Detection Pipeline Overview

1. Load microscopy image
2. Optional structural edge+green+yellow gating
3. Strict HSV yellow-range centroids
4. Merge gate + HSV candidates
5. Non-max suppression (candidate dedupe)
6. Center crop around each candidate (224×224)
7. ResNet-18 scoring
8. Thresholding
9. Optional snap-to-yellow centroid refinement
10. Post-snap NMS
11. Tight-pair pruning
12. Output overlay, CSV, and PNT files

---

## Output Files

Given `sample.png`, the pipeline produces:

- sample_overlay.png # detections drawn

- sample_detections.csv # x,y,score

- sample_artifacts.csv # debug outputs

- sample.pnt # DotDotGoose annotation file

---

## Dependencies

### Runtime:

- Python 3.8+
- NumPy
- OpenCV
- PyTorch
- torchvision
- Pillow
---

### Model Weights

The trained model file used by the AI:

***CellCounter499/ai_model/cell_classifier_best.pth***

This file is bundled into the executable at build time and automatically loaded inside the DotDotGoose UI.  
Users do **not** need to manually interact with it.

## Installation
```bash
git clone https://github.com/.../CellCounter499
python3 -m venv cc-env
source cc-env/bin/activate       # macOS / Linux
# OR on Windows:
cc-env\Scripts\activate
pip install -r requirements.txt
```

## Building the Executable

This builds the DotDotGoose GUI annotation tool used for manual point labeling and reviewing AI-generated outputs.

```bash
venv\Scripts\activate #if using environment
cd package
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --clean --noconfirm ddg.spec #OR 'ddg_mac.spec' for MacOS
```

## Executable Directory: 

CellCounter499/package/dist/ddg

## Running Automated Detection (AI Inside DotDotGoose)

CellCounter inference is integrated directly into the DotDotGoose UI.

There is **no need to run any Python scripts manually**.

Steps:

1. Launch the DotDotGoose executable
2. Open your microscopy image
3. Click:  
   **Automatically Detect Cells**
4. The AI model will run internally and produce detections
5. The detections appear in real-time as cell markers
6. You may delete/add/edit detections manually afterward

This enables a GUI-based human-in-the-loop workflow with no command-line usage.


## Original DotDotGoose:

https://github.com/persts/DotDotGoose