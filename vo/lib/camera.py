import glob
import os

import cv2 as cv
import numpy as np

from .utils import *


class Camera:
    def __init__(
        self, config: dict, stereo: bool, rot_from_dl: bool = False, pose: bool = True
    ) -> None:
        self.idx = 0
        self.stereo = stereo
        self.rot_from_dl = rot_from_dl
        self.seq_num = config["KITTI"]["SEQ"]
        self.pose = pose
        self.poses = None
        if self.pose:
            with open(
                os.path.join(config["KITTI"]["PATH"], "poses", self.seq_num) + ".txt"
            ) as f:
                self.poses = np.array(
                    list(map(lambda x: x.replace("\n", "").split(" "), f.readlines())),
                    dtype=float,
                ).reshape(-1, 3, 4)

        if rot_from_dl:
            self.model = "".join(config["KITTI"]["ROT_PATH"].split("/")[-2])
            self.epoch = config["KITTI"]["ROT_PATH"].split("_")[-2]
            with open(
                os.path.join(
                    config["KITTI"]["ROT_PATH"], f"{self.seq_num}_{self.epoch}_pred.txt"
                )
            ) as f:
                self.dl_poses = np.array(
                    list(map(lambda x: x.replace("\n", "").split(" "), f.readlines())),
                    dtype=float,
                ).reshape(-1, 3, 4)

        seq = os.path.join(config["KITTI"]["PATH"], self.seq_num)
        calib = os.path.join(seq, "calib.txt")

        with open(calib) as f:
            calibs = np.array(
                list(map(lambda x: x.replace("\n", "").split(" ")[1:], f.readlines())),
                dtype=float,
            )
            self.left_proj_mat, self.right_proj_mat = (
                calibs[0].reshape((3, 4)),
                calibs[1].reshape((3, 4)),
            )
            self.left_K = self.left_proj_mat[:, :3]
            self.baseline = -self.right_proj_mat[0, 3] / self.right_proj_mat[0, 0]
            self.right_K = self.right_proj_mat[:, :3]

        seq_left_path = os.path.join(seq, "image_0")
        seq_right_path = os.path.join(seq, "image_1")
        seq_left = sorted(glob.glob(seq_left_path + "/*.png"))
        seq_right = sorted(glob.glob(seq_right_path + "/*.png"))
        assert len(seq_left) == len(seq_right)
        self.seq = zip(seq_left, seq_right)

    def get_image(self, idx: int, left: bool = True) -> np.ndarray:
        return cv.imread(self.seq[idx][0 if left else 1])

    def get_K(self, left: bool = True) -> np.ndarray:
        return self.left_K if left else self.right_K

    def get_focal(self, left: bool = True) -> np.ndarray:
        return self.left_K[0, 0] if left else self.right_K[0, 0]

    def get_principal_point(self, left: bool = True) -> np.ndarray:
        return (
            (self.left_K[0, 2], self.left_K[1, 2])
            if left
            else (self.right_K[0, 2], self.right_K[1, 2])
        )

    def get_projection_matrix(self, left: bool = True) -> np.ndarray:
        return self.left_proj_mat if left else self.right_proj_mat

    def get_baseline(self) -> float:
        return self.baseline

    def get_scale(self, i: int, j: int) -> float:
        return np.linalg.norm(
            get_3d_from_pose(self.poses[i]) - get_3d_from_pose(self.poses[j])
        )

    def __iter__(self):
        for i, s in enumerate(self.seq):
            if self.rot_from_dl and len(self.dl_poses) <= i:
                break
            rel_rot = get_rot_from_pose(self.poses[i - 1]).T @ get_rot_from_pose(
                self.poses[i]
            ) if self.pose else None
            if self.stereo:
                if self.pose:
                    yield cv.imread(s[0]), cv.imread(s[1]), np.vstack(
                        (self.poses[i], [0, 0, 0, 1])
                    ), self.get_scale(i - 1, i), rel_rot, (
                        self.dl_poses[i] if self.rot_from_dl else None
                    )
                else:
                    yield cv.imread(s[0]), cv.imread(s[1]), 1, (
                        self.dl_poses[i] if self.rot_from_dl else None
                    )
            else:
                yield cv.imread(s), np.vstack((self.poses[i], [0, 0, 0, 1])), s

    def __str__(self) -> str:
        return f"left_K:\n{self.left_K}\nright_K:\n{self.right_K}\nbaseline: {self.baseline}\n"


if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read("config/config.ini")

    c = Camera(config, stereo=True, rot_from_dl=True)
    t = Trajectory2D()

    for l, r, p in c:
        cur_traj = t.update(get_2d_from_pose(p), get_2d_from_pose(p))
        cv.imshow("traj", cur_traj)
        cv.waitKey(0)
