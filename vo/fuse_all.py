import configparser
import glob
import shutil

from lib.utils import *
from lib.camera import Camera
from lib.detector import *
from lib.matcher import *
from lib.VO import *

config = configparser.ConfigParser()
config.read("config/config.ini")

model = True
plot_gt = False
seqs = ["0" + str(i) if i < 10 else str(i) for i in range(0, 22)]

ROT_PATH = config["KITTI"]["ROT_PATH"]
for seq in seqs:
    config["KITTI"]["SEQ"] = seq
    PATH = f"{ROT_PATH}/{seq}_*_pred.txt"
    print(PATH)
    for e in sorted(glob.glob(PATH, recursive=True)):
        epoch = e.split("/")[-1].split("_")[1]
        folder_name = e.split(".")[0]
        file_name = e.split("/")[-1]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            continue
        shutil.copyfile(e, os.path.join(folder_name, file_name))
        config["KITTI"]["ROT_PATH"] = folder_name
        vo = VO2D2D(
            Camera(config, stereo=True, rot_from_dl=True, pose=plot_gt),
            SIFTDetector(),
            FLANNMatcher(),
            Trajectory2D(config),
            max_len=3,
            true_rotation=False,
            enable_pose_graph=False,
            pose_graph_only_rot=False,
            show_traj=False,
        )
        shortcut(config, "2d2d", str(vo), gt_pose=plot_gt)
        vo = VO3D2D(
            Camera(config, stereo=True, rot_from_dl=True, pose=plot_gt),
            SIFTDetector(),
            FLANNMatcher(),
            Trajectory2D(config),
            max_len=3,
            true_rotation=False,
            enable_pose_graph=False,
            pose_graph_only_rot=False,
            show_traj=False,
        )

        shortcut(config, "3d2d", str(vo), gt_pose=plot_gt)