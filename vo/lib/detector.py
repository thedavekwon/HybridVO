import cv2 as cv
import itertools as it
import numpy as np

from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import List, Tuple


class Detector(ABC):
    def __init__(self, detector) -> None:
        self.detector = detector

    @abstractmethod
    def detectAndCompute(self, img, mask=None) -> Tuple[np.ndarray, np.ndarray]:
        pass


class CVDetector(Detector):
    def __init__(self, detector, ) -> None:
        super().__init__(detector)

    def detectAndCompute(self, img, mask=None) -> Tuple[np.ndarray, np.ndarray]:
        kps, ds = self.detector.detectAndCompute(img, mask)
        return np.array(kps), ds


class ORBDetector(CVDetector):
    def __init__(self, nfeatures=2000) -> None:
        super().__init__(
            detector=cv.ORB_create(
                nfeatures=nfeatures,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=31,
                fastThreshold=20,
            )
        )

    def __str__(self) -> str:
        return "orb"


class SIFTDetector(CVDetector):
    def __init__(sel, nfeatures=2000) -> None:
        super().__init__(
            detector=cv.SIFT_create(
                nfeatures=nfeatures, #1000
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6,
            )
        )

    def __str__(self) -> str:
        return "sift"


# https://github.com/opencv/opencv/blob/master/samples/python/asift.py
class ASIFTDetector(Detector):
    def __init__(self, detector) -> None:
        super().__init__(detector)
        self.pool = ThreadPool(processes=cv.getNumberOfCPUs())

    def affine_skew(self, tilt, phi, img, mask=None):
        """
        affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
        Ai - is an affine transform matrix from skew_img to img
        """
        h, w = img.shape[:2]
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
        A = np.float32([[1, 0, 0], [0, 1, 0]])
        if phi != 0.0:
            phi = np.deg2rad(phi)
            s, c = np.sin(phi), np.cos(phi)
            A = np.float32([[c, -s], [s, c]])
            corners = [[0, 0], [w, 0], [w, h], [0, h]]
            tcorners = np.int32(np.dot(corners, A.T))
            x, y, w, h = cv.boundingRect(tcorners.reshape(1, -1, 2))
            A = np.hstack([A, [[-x], [-y]]])
            img = cv.warpAffine(
                img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )
        if tilt != 1.0:
            s = 0.8 * np.sqrt(tilt * tilt - 1)
            img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
            img = cv.resize(
                img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv.INTER_NEAREST
            )
            A[0] /= tilt
        if phi != 0.0 or tilt != 1.0:
            h, w = img.shape[:2]
            mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
        Ai = cv.invertAffineTransform(A)
        return img, mask, Ai

    def detectAndCompute(self, img) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        params = [(1.0, 0.0)]
        for t in 2 ** (0.5 * np.arange(1, 6)):
            for phi in np.arange(0, 180, 72.0 / t):
                params.append((t, phi))

        def f(p):
            t, phi = p
            timg, tmask, Ai = self.affine_skew(t, phi, img)
            keypoints, descrs = self.detector.detectAndCompute(timg, tmask)
            for kp in keypoints:
                x, y = kp.pt
                kp.pt = tuple(np.dot(Ai, (x, y, 1)))
            if descrs is None:
                descrs = []
            return keypoints, descrs

        keypoints, descrs = [], []
        if self.pool is None:
            ires = it.imap(f, params)
        else:
            ires = self.pool.imap(f, params)

        for k, d in ires:
            keypoints.extend(k)
            descrs.extend(d)

        return keypoints, np.array(descrs)
