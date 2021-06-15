import configparser

from lib.utils import *
from lib.camera import Camera
from lib.detector import *
from lib.matcher import *
from lib.VO import *

config = configparser.ConfigParser()
config.read("config/config.ini")

vo = VO2D2D(
    Camera(config, stereo=True, rot_from_dl=False, pose=False),
    SIFTDetector(),
    FLANNMatcher(),
    Trajectory2D(config),
    max_len=3,
    true_rotation=False,
    enable_pose_graph=False,
    pose_graph_only_rot=False,
    show_traj=False,
)
print(vo)
vo.run()

vo = VO2D2D(
    Camera(config, stereo=True, rot_from_dl=False, pose=False),
    SIFTDetector(),
    FLANNMatcher(),
    Trajectory2D(config),
    max_len=3,
    true_rotation=False,
    enable_pose_graph=False,
    pose_graph_only_rot=False,
    show_traj=False,
)
print(vo)
vo.run()
