## MultiSenseSeg
Q. Wang, W. Chen, Z. Huang, H. Tang and L. Yang, "MultiSenseSeg: A Cost-Effective Unified Multimodal Semantic Segmentation Model for Remote Sensing," in *IEEE Transactions on Geoscience and Remote Sensing*, vol. 62, pp. 1-24, 2024, Art no. 4703724, doi: [10.1109/TGRS.2024.3390750](https://doi.org/10.1109/TGRS.2024.3390750)

------------------------------
## How to use:
### Quick Start with Sample Data
1. Clone the repository with sample data:
```bash
git clone -b EXP https://github.com/W-qp/MultiSenseSeg.git
```
2. Navigate to the project directory and set up the environment:
```bash
cd MultiSenseSeg
```
 - Please ensure you have PyTorch and other required dependencies installed  
 - Key dependencies: torch, numpy, PIL, tifffile, opencv-python  
 - Use `conda activate <your env name>` to activate your environment.  
3. Run the training script:
```bash
python train.py
```
4. Run the evaluation script:
```bash
python test.py
```
5. Run the prediction script:
```bash
python predict.py
```
### Project Structure
```
dataset/
├── train/
│   ├── modal1/        # First modality
│   ├── modal2/        # Second modality
│   └── gray_label/    # Ground truth labels
├── val/
│   ├── modal1/
│   ├── modal2/
│   └── gray_label/
└── test/
    ├── modal1/
    ├── modal2/
    └── gray_label/

json/
├── classes.json    # Class definitions and RGB color mappings
├── train.json      # Training dataset configuration
├── test.json       # Testing dataset configuration
└── predict.json    # Prediction configuration
```

### Training with Custom Dataset
1. Data Preparation:
   - Place your dataset in the corresponding directories under the `dataset` folder
   - Use `labels_RGB2gray.py` to convert RGB labels to index format
   - Use `crop.py` to crop your images if needed
   - <span style="color: blue">**Important**:</span> Ensure proper value ranges for each modality to guarantee correct normalization during training

2. <span style="color: red">**Configuration**:</span>
   - Modify `json/classes.json` to define your semantic classes and their RGB color mappings
   - Update paths in `json/train.json` for your training data
   - Modify the `--in_chans` parameter in `train.py` according to your modality channels (*e.g.*, using "3, 1" for 3-channel RGB and single-channel DSM data)
   - For Windows users, if you encounter multiprocessing-related errors, please set the num_workers of DataLoader to 0.
   - Consider tuning other hyperparameters in `train.py` to achieve better performance.

3. Start Training:
```bash
python train.py --in_chans "your_channel_numbers"
```

4. Model Evaluation:
```bash
python test.py --in_chans "your_channel_numbers"
```

5. Prediction and Visualization:
```bash
python predict.py --in_chans "your_channel_numbers"
```

### Configuration Files
- `classes.json`: Maps semantic classes to RGB values
```json
{
    "Impervious surfaces": [255, 255, 255],
    "Building": [0, 0, 255],
    ...
}
```
- `train.json`: Specifies paths for training data and results
- `test.json`: Configures paths for accuracy evaluation
- `predict.json`: Sets up paths for inference and visualization

### Citation
```
@ARTICLE{10504922,
  author={Wang, Qingpeng and Chen, Wei and Huang, Zhou and Tang, Hongzhao and Yang, Lan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MultiSenseSeg: A Cost-Effective Unified Multimodal Semantic Segmentation Model for Remote Sensing}, 
  year={2024},
  volume={62},
  number={},
  pages={1-24},
  keywords={Semantic segmentation;Feature extraction;Remote sensing;Data models;Computational modeling;Costs;Semantics;Deep learning;low-cost increment;multimodal;remote sensing;semantic segmentation},
  doi={10.1109/TGRS.2024.3390750}}
```

---
*This README is generated with the assistance of [Cursor](https://cursor.sh/).*
