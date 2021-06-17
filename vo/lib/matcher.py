import cv2 as cv
import numpy as np

from abc import ABC, abstractmethod
from typing import List


class Matcher(ABC):
    @abstractmethod
    def match(self) -> List[cv.DMatch]:
        pass

    @staticmethod
    def show_matches(
        img1: np.ndarray,
        kps1: np.ndarray,
        img2: np.ndarray,
        kps2: np.ndarray,
        matches: List[cv.DMatch],
        sec: int = 1,
    ) -> None:
        match_img = cv.drawMatches(img1, kps1, img2, kps2, matches, None)
        cv.imshow("matched", match_img)
        cv.waitKey(sec)


class BFMatcher(Matcher):
    def __init__(self, orb=False) -> None:
        if orb:
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv.BFMatcher()

    def match(self, ds1: np.ndarray, ds2: np.ndarray) -> List[cv.DMatch]:
        matches = self.matcher.match(ds1, ds2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def __str__(self) -> str:
        return "bf"


class FLANNMatcher(Matcher):
    def __init__(self) -> None:
        self.matcher = cv.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5}, {"checks": 50}
        )

    def match(self, ds1: np.ndarray, ds2: np.ndarray) -> List[cv.DMatch]:
        matches = self.matcher.knnMatch(ds1, ds2, k=2)
        # Distance Ratio Test
        final_matches = []
        for m, n in matches:
            if m.distance <= 0.75 * n.distance:
                final_matches.append(m)
        final_matches = sorted(final_matches, key=lambda m: m.distance)
        return final_matches

    def __str__(self) -> str:
        return "flann"
