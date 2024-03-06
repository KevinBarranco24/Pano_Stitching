import cv2
import numpy as np
import math
from src.images import Image
from src.matching import homography_dlt



class PairMatch:
    def __init__(self, image_a: Image, image_b: Image, matches: list | None = None) -> None:
        """
        Create a new PairMatch object.

        Args:
            image_a: First image of the pair
            image_b: Second image of the pair
            matches: List of matches between image_a and image_b
        """
        self.image_a = image_a
        self.image_b = image_b
        self.matches = matches
        self.H = None
        self.status = None
        self.overlap = None
        self.area_overlap = None
        self._Iab = None
        self._Iba = None
        self.matchpoints_a = None
        self.matchpoints_b = None
        self._inliers = None

    def cams(self):
        return [self.image_b, self.image_a]
    

    def compute_homography(
        self, ransac_reproj_thresh: float = 5, ransac_max_iter: int = 500
    ) -> None:
        """
        Compute the homography between the two images of the pair.

        Args:
            ransac_reproj_thresh: reprojection threshold used in the RANSAC algorithm
            ransac_max_iter: number of maximum iterations for the RANSAC algorithm
        """
        self.matchpoints_a = np.float32(
            [self.image_a.keypoints[match.queryIdx].pt for match in self.matches]
        )
        self.matchpoints_b = np.float32(
            [self.image_b.keypoints[match.trainIdx].pt for match in self.matches]
        )

        self.H, self.status = cv2.findHomography(
            self.matchpoints_b,
            self.matchpoints_a,
            cv2.RANSAC,
            ransac_reproj_thresh,
            maxIters=ransac_max_iter,
        )

    def set_overlap(self) -> None:
        """Compute and set the overlap region between the two images."""
        if self.H is None:
            self.compute_homography()
            self._inliers = self.homography_ransac()
           


        mask_a = np.ones_like(self.image_a.image[:, :, 0], dtype=np.uint8)
        mask_b = cv2.warpPerspective(
            np.ones_like(self.image_b.image[:, :, 0], dtype=np.uint8), self.H, mask_a.shape[::-1]
        )

        self.overlap = mask_a * mask_b
        self.area_overlap = self.overlap.sum()

    def is_valid(self, alpha: float = 8, beta: float = 0.3) -> bool:
        """
        Check if the pair match is valid (i.e. if there are enough inliers with regard to the overlap region).

        Args:
            alpha: alpha parameter used in the comparison
            beta: beta parameter used in the comparison

        Returns:
            valid: True if the pair match is valid, False otherwise
        """
        if self.overlap is None:
            self.set_overlap()

        if self.status is None:
            self.compute_homography()
            self._inliers = self.homography_ransac()

        matches_in_overlap = self.matchpoints_a[
            self.overlap[
                self.matchpoints_a[:, 1].astype(np.int64),
                self.matchpoints_a[:, 0].astype(np.int64),
            ]
            == 1
        ]
        

        return self.status.sum() > alpha + beta * matches_in_overlap.shape[0]

    def contains(self, image: Image) -> bool:
        """
        Check if the given image is contained in the pair match.

        Args:
            image: Image to check

        Returns:
            True if the given image is contained in the pair match, False otherwise
        """
        return self.image_a == image or self.image_b == image

    @property
    def Iab(self):
        if self._Iab is None:
            self.set_intensities()
        return self._Iab

    @Iab.setter
    def Iab(self, Iab):
        self._Iab = Iab

    @property
    def Iba(self):
        if self._Iba is None:
            self.set_intensities()
        return self._Iba

    @Iba.setter
    def Iba(self, Iba):
        self._Iba = Iba

    def set_intensities(self) -> None:
        """
        Compute the intensities of the two images in the overlap region.
        Used for the gain compensation calculation.
        """
        if self.overlap is None:
            self.set_overlap()

        inverse_overlap = cv2.warpPerspective(
            self.overlap, np.linalg.inv(self.H), self.image_b.image.shape[1::-1]
        )

        if self.overlap.sum() == 0:
            print(self.image_a.path, self.image_b.path)

        self._Iab = (
            np.sum(
                self.image_a.image * np.repeat(self.overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / self.overlap.sum()
        )
        self._Iba = (
            np.sum(
                self.image_b.image * np.repeat(inverse_overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / inverse_overlap.sum()
        )

    def estimate_focal_from_homography(self):
        h = self.H

        f1 = None
        f0 = None

        d1 = h[2][0] * h[2][1]
        d2 = (h[2][1] - h[2][0]) * (h[2][1] + h[2][0])
        v1 = -(h[0][0] * h[0][1] + h[1][0] * h[1][1]) / d1
        v2 = (h[0][0] * h[0][0] + h[1][0] * h[1][0] - h[0][1] * h[0][1] - h[1][1] * h[1][1]) / d2
        if (v1 < v2):
            temp = v1
            v1 = v2
            v2 = temp
        if (v1 > 0 and v2 > 0):
            f1 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
        elif (v1 > 0):
            f1 = math.sqrt(v1)
        else:
            return 0

        d1 = h[0][0] * h[1][0] + h[0][1] * h[1][1]
        d2 = h[0][0] * h[0][0] + h[0][1] * h[0][1] - h[1][0] * h[1][0] - h[1][1] * h[1][1]
        v1 = -h[0][2] * h[1][2] / d1
        v2 = (h[1][2] * h[1][2] - h[0][2] * h[0][2]) / d2
        if (v1 < v2):
            temp = v1
            v1 = v2
            v2 = temp
        if (v1 > 0 and v2 > 0):
            f0 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
        elif (v1 > 0):
            f0 = math.sqrt(v1)
        else:
            return 0

        if (math.isinf(f1) or math.isinf(f0)):
            return 0

        return math.sqrt(f1 * f0)
    
    def homography_ransac(self, threshold: float = 5, n_iterations: int = 500):
        """
        Find a best guess for the homography to map pts2 onto 
        the plane of pts1
        
        Input: 
            pts1 - Destination plane
            pts2 - Points to transform onto destination plane#

        Output: Tuple of (Homography projecting pts2 onto pts1 plane, RANSAC inlier kps)
        """

        # Store maxInliners are points, if there is a tie in max, 
        # take the H that has maxInliners with the smallest standard deviation
        maxInliers = []
        bestH = None
        pts1 = self.matchpoints_a
        pts2 = self.matchpoints_b
        

        for i in range(n_iterations):
            # 4 random point indexes
            random_pts_idxs = np.random.choice(len(pts1), 4)

            # Get random sample using random indexes
            pts1_sample = pts1[random_pts_idxs]
            pts2_sample = pts2[random_pts_idxs]

            # Compute H using DLT
            H = homography_dlt.direct_linear_transform(pts1_sample, pts2_sample)

            inliers = []

            # For each correspondence
            for i in range(len(pts1)):
                # Get distance for each correspondance
                distance = self.transfer_error(pts1[i], pts2[i], H)

                # Add correspondence to inliners if distance less than threshold
                if (distance < threshold):
                    inliers.append([pts1[i], pts2[i]])

            
            # If inliers > maxInliers, set as new best H
            if (len(inliers) > len(maxInliers)):
                maxInliers = inliers
                # TODO: else if inliers == maxInliers, pick best H based on smallest standard deviation
        for i in range(len(inliers)):
          maxInliers[i][0], maxInliers[i][1] = maxInliers[i][1], maxInliers[i][0]
        
        return maxInliers  
    
    def transfer_error(self, pt1, pt2, H):
        """
        Calculate the transfer error between the two 

        NOTE: There are other distance metrics (e.g. symmetric distance, could they be better?)

        Input:
            pt1 - Destination point
            pt2 - Point to transform onto destination plane
            H - homography to transform 

        Output: Sum squared distance error
        """

        pt1 = np.append(pt1, [1])
        pt2 = np.append(pt2, [1])

        pt1_projected = ((H @ pt1).T).T 

        diff = (pt2 - pt1_projected)

        sse = np.sum(diff**2)

        return sse    