# Object Detection with Multi-scale Default Boxes

This project implements a simplified SSD-style object detector for the CMPT 742 visual computing assignment. The detector uses multi-scale default boxes to predict object categories and bounding boxes for cats, dogs, and people, with background as the fourth class.

## Project Structure

```text
Object-detection/
+-- README.md
+-- .gitignore
+-- Report.docx
+-- assignment_3.pdf
+-- materials/
    +-- main.py              # Training and testing entry point
    +-- model.py             # SSD network and loss definition
    +-- dataset.py           # Dataset loading, default boxes, IoU matching
    +-- utils.py             # Visualization, NMS, and evaluation helpers
    +-- data/
        +-- put data here.txt
```

Expected dataset layout:

```text
materials/data/
+-- train/
|   +-- images/
|   +-- annotations/
+-- test/
    +-- images/
    +-- annotations/
```

Annotation files are expected to be `.txt` files with the same base name as the corresponding image.

## Method Overview

The model follows the Single Shot MultiBox Detector (SSD) idea: instead of proposing regions first, it predicts class confidence and bounding box offsets directly from several feature-map scales.

Default boxes are generated on four prediction layers:

| Feature map | Cells | Default boxes per cell |
| --- | ---: | ---: |
| 10 x 10 | 100 | 4 |
| 5 x 5 | 25 | 4 |
| 3 x 3 | 9 | 4 |
| 1 x 1 | 1 | 4 |

This gives:

```text
4 * (10 * 10 + 5 * 5 + 3 * 3 + 1 * 1) = 540 default boxes
```

For each default box, the network predicts:

- class confidence over 4 classes: cat, dog, person, background
- bounding box regression values

Ground-truth boxes are matched to default boxes using IoU. Boxes above the matching threshold are treated as positive anchors, and the best-matching default box is also selected so each object has at least one assigned anchor.

## Report Summary

According to the project report, the model works best on images containing a single person or a single animal. Person detection is comparatively stable, while cat and dog detection is more difficult, especially when there are multiple animals in one image.

Observed limitations:

- Multiple objects from the same class may be merged into one detection.
- Bounding boxes can become inaccurate when several cats or dogs appear together.
- The model sometimes confuses cats and dogs.
- The best confidence threshold on the selected test images was around `80%`.

Possible improvements discussed in the report:

- Add class weights in `SSD_loss` to improve learning for underperforming classes.
- Add an extra misclassification penalty for wrong class predictions.
- Use `AdamW` with weight decay, for example `weight_decay=1e-2`.
- Add more data augmentation, especially for classes with fewer samples.
- Train with more labeled data.

## Setup

Create and activate a Python environment, then install the main dependencies:

```bash
pip install numpy opencv-python torch torchvision
```

CUDA is used by default in `materials/main.py`, so a CUDA-enabled PyTorch installation is recommended. If running on CPU, update the `.cuda()` calls in the code accordingly.

## Training

From the `materials/` directory:

```bash
python main.py
```

Training uses:

- `materials/data/train/images/`
- `materials/data/train/annotations/`

The script saves model weights as:

```text
materials/network.pth
```

## Testing

From the `materials/` directory:

```bash
python main.py --test
```

Testing uses:

- `materials/data/test/images/`
- `materials/data/test/annotations/`

The script loads:

```text
materials/network.pth
```

## Main Files

`materials/main.py` controls training, validation, testing, optimizer setup, model saving, and visualization.

`materials/model.py` defines the SSD model and the SSD loss function.

`materials/dataset.py` handles default box generation, IoU calculation, anchor matching, image loading, annotation loading, and dataset preparation.

`materials/utils.py` contains prediction visualization utilities, non-maximum suppression placeholders, and evaluation helper placeholders.

## Notes

Large local files such as datasets, trained weights, generated visualizations, cache folders, and IDE metadata are ignored by `.gitignore`. Keep only source code, reports, and small documentation files in version control.
