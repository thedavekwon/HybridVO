import cv2 as cv
import g2o
import numpy as np

from .camera import Camera
from .detector import Detector
from .matcher import Matcher
from .utils import *
from .optimizer import *

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple


class VO(ABC):
    def __init__(
        self,
        cam: Camera,
        detector: Detector,
        matcher: Matcher,
        traj: Trajectory2D,
        max_len: int,
        true_rotation: bool,
        enable_pose_graph: bool,
        pose_graph_only_rot: bool,
        show_traj: bool,
    ) -> None:
        self.cam = cam

        self.detector = detector
        self.matcher = matcher
        self.traj = traj
        self.true_rotation = true_rotation
        self.enable_pose_graph = enable_pose_graph
        self.pose_graph_only_rot = pose_graph_only_rot
        self.show_traj = show_traj
        self.max_len = max_len

        self.cur_R: np.ndarray = np.eye(3)
        self.cur_t: np.ndarray = np.zeros((3, 1))
        self.cache = deque(maxlen=max_len if enable_pose_graph else 2)

    @abstractmethod
    def get_T(self):
        pass

    @abstractmethod
    def get_T_from_cache(self, i, j):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def update(
        self, img: np.ndarray, scale: float, idx: int, gt_pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def triangulate(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> Tuple[
        np.ndarray,  # Triangulated Landmarks
        np.ndarray,  # Left inlier indexes
        np.ndarray,  # Right inlier indexes
        np.ndarray,  # Left Keypoints
        np.ndarray,  # Left Descriptors
        np.ndarray,  # Right Keypoints
        np.ndarray,  # Right Descriptors
        List[cv.DMatch],  # Matches
    ]:
        kps1, ds1 = self.detector.detectAndCompute(left_img, None)
        kps2, ds2 = self.detector.detectAndCompute(right_img, None)
        matches = self.matcher.match(ds1, ds2)

        K = self.cam.get_K()
        T1 = K @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        T2 = K @ np.concatenate(
            (np.eye(3), np.array([-self.cam.get_baseline(), 0, 0]).reshape(3, 1)),
            axis=1,
        )

        pts_px_1, pts_px_2 = [], []
        left_indexes, right_indexes = [], []
        for m in matches:
            pts_px_1.append(kps1[m.queryIdx].pt)
            pts_px_2.append(kps2[m.trainIdx].pt)
            left_indexes.append(m.queryIdx)
            right_indexes.append(m.trainIdx)
        pts_px_1, pts_px_2 = np.array(pts_px_1).T, np.array(pts_px_2).T
        left_indexes, right_indexes = np.array(left_indexes), np.array(right_indexes)
        pts4D = cv.triangulatePoints(T1, T2, pts_px_1, pts_px_2).T
        pts3D = cv.convertPointsFromHomogeneous(pts4D).reshape(-1, 3)

        # Remove landmarks that are too far away or behind the camera
        pts3D_dist = np.linalg.norm(pts3D, axis=1)
        outlier_indexes = np.logical_and(
            pts3D_dist < np.mean(pts3D_dist), pts3D[:, 2] > 0.0
        )
        pts3D = pts3D[outlier_indexes]
        left_indexes = left_indexes[outlier_indexes]
        right_indexes = right_indexes[outlier_indexes]
        return pts3D, left_indexes, right_indexes, kps1, ds1, kps2, ds2, matches

    def __str__(self) -> str:
        if self.true_rotation:
            rot = "gtrot"
        elif self.cam.rot_from_dl:
            rot = f"dlrot_{self.cam.model}_{self.cam.epoch}"
        else:
            rot = "norot"
        fg = (
            (
                f"pg({self.max_len})"
                if not self.pose_graph_only_rot
                else f"rotpg({self.max_len})"
            )
            if self.enable_pose_graph
            else "nopg"
        )
        return f"{self.cam.seq_num}_{rot}_{fg}_{self.detector}_{self.matcher}"

    def pose_graph_g2o(self, idx: int, with_scale: bool = False) -> None:
        graph = PoseGraphOptimizer()

        for i in range(len(self.cache)):
            cur_idx = -len(self.cache) + i
            if self.pose_graph_only_rot:
                graph.add_vertex(
                    id=i,
                    pose=g2o.Isometry3d(self.traj.Rs[cur_idx], np.zeros((3, 1))),
                    fixed=i != len(self.cache) - 1,
                )
            else:
                graph.add_vertex(
                    id=i,
                    pose=g2o.Isometry3d(self.traj.Rs[cur_idx], self.traj.ts[cur_idx]),
                    fixed=i != len(self.cache) - 1,
                )

        for i in range(len(self.cache)):
            for j in range(i + 1, len(self.cache)):
                if j != len(self.cache) - 1:
                    continue
                R, t = self.get_T_from_cache(i, j)
                if with_scale:
                    s = self.cam.get_scale(
                        idx - len(self.cache) + i, idx - len(self.cache) + j
                    )
                    t *= s
                if self.pose_graph_only_rot:
                    graph.add_edge((i, j), g2o.Isometry3d(R, np.zeros((3, 1))))
                else:
                    graph.add_edge((i, j), g2o.Isometry3d(R, t))
        graph.optimize()
        pose = graph.get_pose(len(self.cache) - 1)

        # Rs = [graph.get_pose(i).R for i in range(len(self.cache))]
        # ts = (
        #     [graph.get_pose(i).t.reshape(3, 1) for i in range(len(self.cache))]
        #     if not self.pose_graph_only_rot
        #     else None
        # )

        if self.pose_graph_only_rot:
            Rs = [pose.R]
            self.traj.replace_last(Rs=Rs)
        else:
            Rs = [pose.R]
            ts = [pose.t.reshape(3, 1)]
            self.traj.replace_last(Rs=Rs, ts=ts)

    @abstractmethod
    def save(self):
        pass


class VO2D2D(VO):
    def __init__(
        self,
        cam: Camera,
        detector: Detector,
        matcher: Matcher,
        traj: Trajectory2D,
        max_len: int,
        true_rotation: bool = False,
        enable_pose_graph: bool = False,
        pose_graph_only_rot: bool = False,
        show_traj: bool = False,
    ) -> None:
        super().__init__(
            cam,
            detector,
            matcher,
            traj,
            max_len,
            true_rotation,
            enable_pose_graph,
            pose_graph_only_rot,
            show_traj,
        )

    def __str__(self) -> str:
        return super().__str__() + "_2d2d"

    def save(self, gt_pose=True):
        self.traj.save(str(self), "2d2d", gt_pose=gt_pose)

    def run(self) -> None:
        if self.cam.pose:
            for idx, (l, _, p, s, _, dl_p) in enumerate(self.cam):
                R, t = self.update(l, s, idx, p, dl_p)
                self.traj.update(idx, p, R, t)
                if self.show_traj:
                    self.traj.show(1)
            self.save()
        else:
            for idx, (l, _, s, dl_p) in enumerate(self.cam):
                R, t = self.update(l, s, idx, None, None)
                self.traj.update(idx, None, R, t)
                if self.show_traj:
                    self.traj.show(1)
            self.save(False)

    def get_T_from_cache(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_T(*self.cache[i], *self.cache[j])

    def get_T(
        self, kps1: np.ndarray, ds1: np.ndarray, kps2: np.ndarray, ds2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        matches = self.matcher.match(ds1, ds2)
        pts1 = []
        pts2 = []
        for m in matches:
            pts1.append(kps1[m.queryIdx].pt)
            pts2.append(kps2[m.trainIdx].pt)
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        E, mask = cv.findEssentialMat(
            points1=pts2,
            points2=pts1,
            cameraMatrix=self.cam.get_K(),
            method=cv.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        pts, R, t, mask_pose = cv.recoverPose(
            E=E, points1=pts2, points2=pts1, cameraMatrix=self.cam.get_K(), mask=mask
        )
        return R, t

    def update(
        self,
        img: np.ndarray,
        scale: float,
        idx: int,
        gt_pose: np.ndarray,
        dl_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        kps, ds = self.detector.detectAndCompute(img)
        self.cache.append((kps, ds))
        if idx:
            prev_kps, prev_ds = self.cache[-2]
            R, t = self.get_T(prev_kps, prev_ds, kps, ds)
            if scale > 0.1:
                self.cur_t += scale * self.cur_R @ t
            if self.true_rotation:
                self.cur_R = get_rot_from_pose(gt_pose)
            elif self.cam.rot_from_dl:
                self.cur_R = get_rot_from_pose(dl_pose)
            else:
                self.cur_R = R @ self.cur_R
            self.traj.append(gt_pose, self.cur_R, self.cur_t, R, scale * t)
        # self.traj.append(gt_pose, self.cur_R, self.cur_t)
        if len(self.cache) == self.max_len and self.enable_pose_graph:
            # print("before pose graph")
            # print("R:")
            # print(self.cur_R)
            # print("t:")
            # print(self.cur_t)
            self.pose_graph_g2o(idx, with_scale=False)
            # print("after pose graph")
            # print("R:")
            # print(self.cur_R)
            # print("t:")
            # print(self.cur_t)
        return self.cur_R, self.cur_t


class VO3D2D(VO):
    def __init__(
        self,
        cam: Camera,
        detector: Detector,
        matcher: Matcher,
        traj: Trajectory2D,
        max_len: int,
        true_rotation: bool = False,
        enable_pose_graph: bool = False,
        pose_graph_only_rot: bool = False,
        show_traj: bool = False,
    ) -> None:
        super().__init__(
            cam,
            detector,
            matcher,
            traj,
            max_len,
            true_rotation,
            enable_pose_graph,
            pose_graph_only_rot,
            show_traj,
        )

    def __str__(self) -> str:
        return super().__str__() + "_3d2d"

    def save(self, gt_pose=True):
        self.traj.save(str(self), "3d2d", gt_pose=gt_pose)

    def run(self):
        if self.cam.pose:
            for idx, (l, r, p, _, _, dl_p) in enumerate(self.cam):
                R, t = self.update((l, r), idx, p, dl_p)
                self.traj.update(idx, p, R, t)
                if self.show_traj:
                    self.traj.show(1)
            self.save()
        else:
            for idx, (l, r, _, dl_p) in enumerate(self.cam):
                R, t = self.update((l, r), idx, None, dl_p)
                self.traj.update(idx, None, R, t)
                if self.show_traj:
                    self.traj.show(1)
            self.save(gt_pose=False)

    def update(
        self,
        img: Tuple[np.ndarray, np.ndarray],
        idx: int,
        gt_pose: np.ndarray,
        dl_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_img, right_img = img
        vel = None
        if not idx:
            (
                tpts,
                left_indexes,
                right_indexes,
                left_kps,
                left_ds,
                right_kps,
                right_ds,
                matches,
            ) = self.triangulate(left_img, right_img)
        else:
            (
                prev_tpts,
                prev_left_indexes,
                _,
                _,
                prev_left_ds,
                _,
                _,
                _,
                _,
                _,
                prev_vel,
            ) = self.cache[-1]

            (
                tpts,
                left_indexes,
                right_indexes,
                left_kps,
                left_ds,
                right_kps,
                right_ds,
                matches,
            ) = self.triangulate(left_img, right_img)

            R, t = self.get_T(
                prev_tpts,
                prev_left_indexes,
                prev_left_ds,
                left_indexes,
                left_kps,
                left_ds,
            )
            vel = np.linalg.norm(t)

            if prev_vel and vel < 4 * prev_vel:
                self.cur_t += self.cur_R @ t
                if self.true_rotation:
                    self.cur_R = get_rot_from_pose(gt_pose)
                elif self.cam.rot_from_dl:
                    self.cur_R = get_rot_from_pose(dl_pose)
                else:
                    self.cur_R = R @ self.cur_R
                self.traj.append(gt_pose, self.cur_R, self.cur_t, R, t)
                if len(self.cache) == self.max_len and self.enable_pose_graph:
                    self.pose_graph_g2o(idx, with_scale=False)
            else:
                vel = prev_vel if prev_vel else vel
                self.traj.append(
                    gt_pose, self.cur_R, self.cur_t, np.eye(3), np.zeros((3, 1))
                )
        self.cache.append(
            (
                tpts,
                left_indexes,
                right_indexes,
                left_kps,
                left_ds,
                right_kps,
                right_ds,
                matches,
                left_img,
                right_img,
                vel,
            )
        )
        return self.cur_R, self.cur_t

    def get_T_from_cache(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        tpts1, left_indexes1, _, _, left_ds1, _, _, _, _, _, _ = self.cache[i]
        _, left_indexes2, _, left_kps2, left_ds2, _, _, _, _, _, _ = self.cache[j]
        return self.get_T(
            tpts1, left_indexes1, left_ds1, left_indexes2, left_kps2, left_ds2
        )

    def get_T(
        self,
        prev_tpts: np.ndarray,
        prev_left_indexes: np.ndarray,
        prev_left_ds: np.ndarray,
        left_indexes: np.ndarray,
        left_kps: np.ndarray,
        left_ds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PnP
        """
        tpts_matches = self.matcher.match(
            prev_left_ds[prev_left_indexes], left_ds[left_indexes]
        )
        tpts = prev_tpts[list(map(lambda m: m.queryIdx, tpts_matches))]
        pts = np.array(
            list(map(lambda m: left_kps[left_indexes][m.trainIdx].pt, tpts_matches))
        )
        _, rvec, t, _ = cv.solvePnPRansac(
            objectPoints=tpts,
            imagePoints=pts,
            cameraMatrix=self.cam.get_K(),
            distCoeffs=None,
            iterationsCount=1000,
            reprojectionError=1.0,
            confidence=0.999,
            flags=cv.SOLVEPNP_AP3P,
        )
        R, _ = cv.Rodrigues(rvec)
        R = R.T
        t = -R @ t
        return R, t


class VO3D3D(VO):
    def __init__(
        self,
        cam: Camera,
        detector: Detector,
        matcher: Matcher,
        traj: Trajectory2D,
        max_len: int,
        true_rotation: bool = False,
        enable_pose_graph: bool = False,
        pose_graph_only_rot: bool = False,
        show_traj: bool = False,
    ) -> None:
        super().__init__(
            cam,
            detector,
            matcher,
            traj,
            max_len,
            true_rotation,
            enable_pose_graph,
            pose_graph_only_rot,
            show_traj,
        )
        self.optimize = optimize

    def __str__(self) -> str:
        return super().__str__() + "_3d3d"

    def save(self):
        self.traj.save(str(self), "3d3d")

    def run(self) -> None:
        for idx, (l, r, p, _, rel_rot) in enumerate(self.cam):
            R, t = self.update((l, r), idx, p, rel_rot)
            self.traj.update(idx, p, R, t)
            self.traj.show()
        self.save()

    def get_T(
        self, prev_tpts: np.ndarray, tpts: np.ndarray, gt_rel_rot: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ICP
        """
        # Center of mass
        prev_com, com = (
            np.mean(prev_tpts, axis=0).flatten().reshape(1, 3),
            np.mean(tpts, axis=0).flatten().reshape(1, 3),
        )

        if not self.true_rotation:
            prev_tpts -= prev_com
            tpts -= com
            W = prev_tpts.T @ tpts
            U, _, V = np.linalg.svd(W)
            R = (U @ V).T
            if np.linalg.det(R) < 0:
                R = -R
        else:
            R = gt_rel_rot
        t = com.T - R @ prev_com.T
        return R, t

    def update(
        self,
        img: Tuple[np.ndarray, np.ndarray],
        idx: int,
        gt_pose: np.ndarray,
        gt_rel_rot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_img, right_img = img
        if not idx:
            (
                tpts,
                left_indexes,
                right_indexes,
                left_kps,
                left_ds,
                right_kps,
                right_ds,
                matches,
            ) = self.triangulate(left_img, right_img)
            self.cache.append(
                (
                    tpts,
                    left_indexes,
                    right_indexes,
                    left_kps,
                    left_ds,
                    right_kps,
                    right_ds,
                    matches,
                    left_img,
                    right_img,
                )
            )
            self.traj.append(gt_pose, self.cur_R, self.cur_t)
        else:
            (
                prev_tpts,
                prev_left_indexes,
                _,
                _,
                prev_left_ds,
                _,
                _,
                _,
                _,
                _,
            ) = self.cache[-1]

            (
                tpts,
                left_indexes,
                right_indexes,
                left_kps,
                left_ds,
                right_kps,
                right_ds,
                matches,
            ) = self.triangulate(left_img, right_img)
            self.cache.append(
                (
                    tpts,
                    left_indexes,
                    right_indexes,
                    left_kps,
                    left_ds,
                    right_kps,
                    right_ds,
                    matches,
                    left_img,
                    right_img,
                )
            )

            tpts_matches = self.matcher.match(
                prev_left_ds[prev_left_indexes], left_ds[left_indexes]
            )

            # Show current match
            # self.traj.cur_show(
            #     (prev_left_img, prev_left_kps, prev_left_indexes),
            #     (prev_right_img, prev_right_kps, prev_right_indexes),
            #     (left_img, left_kps, left_indexes),
            #     (right_img, right_kps, right_indexes),
            #     prev_matches,
            #     matches,
            #     tpts_matches,
            # )

            prev_final_tpts = prev_tpts[list(map(lambda m: m.queryIdx, tpts_matches))]
            final_tpts = tpts[list(map(lambda m: m.trainIdx, tpts_matches))]

            R, t = self.get_T(
                prev_tpts=prev_final_tpts, tpts=final_tpts, gt_rel_rot=gt_rel_rot
            )
            R = R.T
            t = -R @ t
            self.cur_t += self.cur_R @ t
            self.cur_R = (
                get_rot_from_pose(gt_pose) if self.true_rotation else R @ self.cur_R
            )
            self.traj.append(gt_pose, self.cur_R, self.cur_t)
        return self.cur_R, self.cur_t
