# Multitask System for Exercise recognition and Counting
This is the official implementation of MTI2021 paper:["Deep Learning-Enabled Multitask System for Exercise Recognition and Counting"](https://www.mdpi.com/2414-4088/5/9/55)

The code implementation refers to the project["Deep Human Action Recognition"](https://github.com/dluvizon/deephar).

If you want to know more about machine learning-based exercise field, please refer to our survey: ["Digital Twin Coaching for Physical Activities: A Survey"](https://www.mdpi.com/1424-8220/20/20/5936)

# System Overview
The inputs are RGB frames from an exercise video. The whole system is mainly composed of 4 parts: MSPN 2D human pose estimation model, joint location calculation, heatmap processing and the multitask model for exercise recognition & counting.
![](/images/system.JPG)

# Multitask Model
![](/images/model.JPG)
# Requirements
- Tensorflow 1.16
- Python 3
# Dataset Preparation
Rep-Penn Dataset is not provided here. If you want to create the dataset in the same way, please refer to our paper.

The optional method is generating a heatmap for one-cycle exercise videos, and duplicate&concatenate heatmaps using similar methods introduced in the paper.
# Running the code
## Training
Train from scratch. Please change the keywords ('action' or 'counting') to train corresponding branch. 
```
python3 train_multitask.py
```

## Testing
```
python3 eval_multitask.py
```

# Citation
If you use this code, please cite the following:
```
@article{yu2021deep,
  title={Deep Learning-Enabled Multitask System for Exercise Recognition and Counting},
  author={Yu, Qingtian and Wang, Haopeng and Laamarti, Fedwa and El Saddik, Abdulmotaleb},
  journal={Multimodal Technologies and Interaction},
  volume={5},
  number={9},
  pages={55},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

