import pickle
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional


def get_3d_from_pose(pose: np.ndarray) -> np.ndarray:
    return pose[:3, 3]  # XZY


def get_2d_from_pose(pose: np.ndarray) -> np.ndarray:
    return get_3d_from_pose(pose)[[0, 2]]


def get_rot_from_pose(pose: np.ndarray) -> np.ndarray:
    return pose[:3, :3]


def get_t_xy(t: np.ndarray) -> np.ndarray:
    return t[[0, 2]]


class Trajectory2D:
    def __init__(self, config) -> None:
        self.config = config
        self.traj = np.zeros((1000, 1000, 3), dtype=np.uint8)
        self.errors: List[float] = []
        self.Rs: List[np.ndarray] = [np.eye(3)]
        self.ts: List[np.ndarray] = [np.zeros((3, 1))]
        self.rel_Rs: List[np.ndarray] = []
        self.rel_ts: List[np.ndarray] = []
        self.ps: List[np.ndarray] = []
        self.trajs: List[np.ndarray] = []
        self.est_xys: List[Tuple[float, float]] = []
        self.gt_xys: List[Tuple[float, float]] = []

    def append(
        self,
        p: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        rel_R: np.ndarray = None,
        rel_t: np.ndarray = None,
    ):
        if p is not None:
            p = p.copy()
            self.ps.append(p)  # Pose
        
        R = R.copy()
        t = t.copy()
        self.Rs.append(R)  # Rotation
        self.ts.append(t)  # translation
        self.trajs.append(np.concatenate((R, t), axis=1).flatten())

        if rel_R is not None:
            self.rel_Rs.append(rel_R.flatten())
        if rel_t is not None:
            self.rel_ts.append(rel_t.flatten())

    def replace_last(self, Rs: Optional[np.ndarray], ts: Optional[np.ndarray] = None):
        for i in range(len(Rs)):
            cur_idx = len(Rs) - i - 1
            if Rs is not None:
                self.Rs[cur_idx] = Rs[i]
            if ts is not None:
                self.ts[cur_idx] = ts[i]
        if Rs is not None:
            self.cur_R = Rs[-1]
        if ts is not None:
            self.cur_t = ts[-1]

    def update(
        self,
        idx: int,
        p: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        output: bool = True,
    ) -> None:
        adj_x, adj_y = 500, 500
        if self.config["KITTI"]["SEQ"] == "01":
            adj_x += -450
            adj_y += 450
        est_xy = get_t_xy(t).flatten()
        gt_xy = get_2d_from_pose(p) if p is not None else None

        self.est_xys.append((est_xy[0], est_xy[1]))
        if p is not None:
            self.gt_xys.append((gt_xy[0], gt_xy[1]))

        est_x, est_y = int(est_xy[0]) + adj_x, int(est_xy[1]) + adj_y
        if p is not None:
            gt_x, gt_y = int(gt_xy[0]) + adj_x, int(gt_xy[1]) + adj_y
        if p is not None:
            self.errors.append(np.linalg.norm(est_xy - gt_xy))
            if output and idx and idx % 500 == 0:
                print(idx, np.mean(self.errors))

        cv.circle(self.traj, (est_x, est_y), 1, (0, 0, 255), 1)
        if p is not None:
            cv.circle(self.traj, (gt_x, gt_y), 1, (255, 0, 0), 1)

    def cur_show(
        self,
        l1: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        r1: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        l2: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        r2: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        m1: List[cv.DMatch] = None,
        m2: List[cv.DMatch] = None,
        m12: List[cv.DMatch] = None,
    ) -> None:
        if any([l1 is not None, r1 is not None, l2 is not None, r2 is not None]):
            prev_cur = cv.drawMatches(
                l1[0], l1[1][l1[2]], l2[0], l2[1][l2[2]], m12, None
            )
            cv.imshow("prev_cur", prev_cur)
        elif l1 is not None and l2 is not None:
            prev_cur = cv.drawMatches(*l1, *l2, m12, None)
            cv.imshow("prev_cur", prev_cur)
        cv.waitKey(1)

    def show(self, sec: int = 1):
        cv.imshow("traj", self.traj)
        cv.waitKey(sec)

    def plot(self) -> None:
        self.plot_traj()
        self.plot_gt()
        plt.show()

    def plot_traj(self) -> None:
        est_xys = np.array(self.est_xys)
        plt.scatter(est_xys[:, 0], est_xys[:, 1], s=0.1, c="r")

    def plot_gt(self) -> None:
        gt_xys = np.array(self.gt_xys)
        plt.scatter(gt_xys[:, 0], gt_xys[:, 1], s=0.1, c="b")

    def save(self, filename, postfix, gt_pose=True) -> None:
        with open(f"pickles/{filename}.pkl", "wb") as f:
            pickle.dump(self, f)
        self.plot_traj()
        if gt_pose:
            self.plot_gt()
        plt.legend(["Ours", "Ground Truth"])
        plt.savefig(f"figs/{filename}.png")
        plt.clf()
        if postfix:
            self.trajs = []
            for R, t in zip(self.Rs, self.ts):
                self.trajs.append(np.concatenate((R, t), axis=1).flatten())
        np.savetxt(f"results/{filename}_pred.txt", self.trajs, fmt="%1.8f")
        seq_num = self.config["KITTI"]["SEQ"]
        if postfix:
            np.savetxt(f"rots/{seq_num}_{postfix}.txt", self.rel_Rs, fmt="%1.8f")
            np.savetxt(f"trans/{seq_num}_{postfix}.txt", self.rel_ts, fmt="%1.8f")

    def error(self) -> float:
        return np.mean(self.errors)


def shortcut(config, type, filename, gt_pose=True):
    seq_num = config["KITTI"]["SEQ"]
    epoch = config["KITTI"]["ROT_PATH"].split("_")[-2]
    with open(
        os.path.join(config["KITTI"]["REL_TRANS_PATH"], f"{seq_num}_{type}.txt")
    ) as f:
        trans = np.array(
            list(map(lambda x: x.replace("\n", "").split(" "), f.readlines())),
            dtype=float,
        ).reshape(-1, 3, 1)
    with open(
        os.path.join(config["KITTI"]["ROT_PATH"], f"{seq_num}_{epoch}_pred.txt")
    ) as f:
        dl_poses = np.array(
            list(map(lambda x: x.replace("\n", "").split(" "), f.readlines())),
            dtype=float,
        ).reshape(-1, 3, 4)

    if gt_pose:
        with open(os.path.join(config["KITTI"]["PATH"], "poses", seq_num) + ".txt") as f:
            poses = np.array(
                list(map(lambda x: x.replace("\n", "").split(" "), f.readlines())),
                dtype=float,
            ).reshape(-1, 3, 4)

    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))

    traj = Trajectory2D(config)

    if gt_pose:
        for idx, (t, dl_p, p) in enumerate(zip(trans, dl_poses, poses)):
            cur_t += cur_R @ t
            cur_R = get_rot_from_pose(dl_p)
            traj.append(p, cur_R.copy(), cur_t.copy(), None, None)
            traj.update(idx, p, cur_R.copy(), cur_t.copy())
            # traj.show()
    else:
        for idx, (t, dl_p) in enumerate(zip(trans, dl_poses)):
            cur_t += cur_R @ t
            cur_R = get_rot_from_pose(dl_p)
            traj.append(None, cur_R.copy(), cur_t.copy(), None, None)
            traj.update(idx, None, cur_R.copy(), cur_t.copy())
            # traj.show()
    traj.save(filename, "", gt_pose=gt_pose)
