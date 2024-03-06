import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation


class Image:
    focal = 1
    ppx = 0
    ppy = 0
    R = None

    
    
    def __init__(self, path: str, size: int | None = None) -> None:
        """
        Image constructor.

        Args:
            path: path to the image
            size: maximum dimension to resize the image to
        """
        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        if size is not None:
            h, w = self.image.shape[:2]
            if max(w, h) > size:
                if w > h:
                    self.image = cv2.resize(self.image, (size, int(h * size / w)))
                else:
                    self.image = cv2.resize(self.image, (int(w * size / h), size))

        self.keypoints = None
        self.features = None
        self.H: np.ndarray = np.eye(3)
        self.component_id: int = 0
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)
        
    @property
    def K(self):
        I = np.identity(3, dtype=np.float64)
        I[0][0] = self.focal
        I[0][2] = self.ppx
        I[1][1] = self.focal
        I[1][2] = self.ppy
        return I

    def compute_features(self) -> None:
        """Compute the features and the keypoints of the image using SIFT."""
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(self.image, None)
        self.keypoints = keypoints
        self.features = features

    def angle_parameterisation(self):
        # rotation = Rotation.from_matrix(self.R)
        # return rotation.as_rotvec()
        u,s,v = np.linalg.svd(self.R)
        R_new = u @ (v) # TODO: might need to be transposed...
        if (np.linalg.det(R_new) < 0):
            R_new *= -1

        # print('')
        rx = R_new[2][1] - R_new[1][2]
        ry = R_new[0][2] - R_new[2][0]
        rz = R_new[1][0] - R_new[0][1]

        s = math.sqrt(rx**2 + ry**2 + rz**2)
        if (s < 1e-7):
            rx, ry, rz = 0, 0, 0
        else:
            cos = (R_new[0][0] + R_new[1][1] + R_new[2][2] - 1) * 0.5
            if (cos > 1):
                cos = 1
            elif (cos < -1):
                cos = -1
        
            theta = np.arccos(cos)
            mul = 1 / s * theta
            rx *= mul
            ry *= mul
            rz *= mul

        return np.array([rx, ry, rz], dtype=np.float64)

    def rotvec_to_matrix(self, rotvec):
        rotation = Rotation.from_rotvec(rotvec)
        return rotation.as_matrix()


