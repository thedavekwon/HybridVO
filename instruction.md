# Instruction
## Dataset
Kitti Dataset: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

Odometry Color for Deep Learning Method

Odometry Black for 3D-to-2D Method

Odometry Ground Truth

Configure Path in src/constants.py

## Configuration
Configure all variables in src/constants.py

Download pretrained [FlowNet](https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5) and configure path in src/constants.py

## Training
```
// DeepVO
python3 src/deepvo.py

// HybridVO
python3 src/hybridvo.py
```

## Testing
```
// Regular
python3 src/draw.py

// 6D Representation
python3 src/draw_rot_6d.py
```

## Evaluating
Use https://github.com/MichaelGrupp/evo or https://github.com/thedavekwon/KITTI_odometry_evaluation_tool for evaluation.

```
\\ Sample evo commands
evo_rpe kitti ground\ truth.txt 3d-to-2d.txt -va --save_results traditional_rpe.zip
evo_rpe kitti ground\ truth.txt DeepVO.txt -va --save_results deepvo_rpe.zip
# evo_rpe kitti ground\ truth.txt Hybrid\ VO.txt -va --save_results ours_rpe.zip
evo_res *_rpe.zip -p --save_table rpe.csv
evo_traj kitti 3d-to-2d.txt DeepVO.txt Hybrid\ VO.txt --ref=ground\ truth.txt -p --plot_mode=xz --save_plot ${SEQ}
```