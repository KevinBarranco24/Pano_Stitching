import cv2

from src.images import Image
from src.matching.pair_match import PairMatch
from src.images import camera


class MultiImageMatches:
    def __init__(self, images, ratio: float = 0.75) -> None:
        """
        Create a new MultiImageMatches object.

        Args:
            images: images to compare
            ratio: ratio used for the Lowe's ratio test
        """
        self.images = images
        self.matches = {image.path: {} for image in images}
        self.ratio = ratio
        self._cameras = [camera.Camera(image) for image in self.images]

    def get_matches(self, image_a: Image, image_b: Image) -> list:
        """
        Get matches for the given images.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            matches: List of matches between the two images
        """
        if image_b.path not in self.matches[image_a.path]:
            matches = self.compute_matches(image_a, image_b)
            self.matches[image_a.path][image_b.path] = matches

        return self.matches[image_a.path][image_b.path]

    def get_pair_matches(self, max_images: int = 6) -> list[PairMatch]:
        """
        Get the pair matches for the given images.

        Args:
            max_images: Number of matches maximum for each image

        Returns:
            pair_matches: List of pair matches
        """
        pair_matches = []
        for i, image_a in enumerate(self.images):
            possible_matches = sorted(
                self.images[:i] + self.images[i + 1 :],
                key=lambda image, ref=image_a: len(self.get_matches(ref, image)),
                reverse=True,
            )[:max_images]
            for image_b in possible_matches:
                if self.images.index(image_b) > i:
                    pair_match = PairMatch(image_a, image_b, self.get_matches(image_a, image_b))
                    if pair_match.is_valid():
                        pair_matches.append(pair_match)
                        self._cameras[i]
                        self._cameras[self.images.index(image_b)]
                        print("self camaras pair match K")
                        print(self._cameras[i].K)
                        print("self camaras pair match R")
                        print(self._cameras[self.images.index(image_a)].R)
                        print("self camaras pair match K")
                        print(self._cameras[i].K)
                        print("self camaras pair match R")
                        print(self._cameras[self.images.index(image_b)].R)
        return pair_matches

    def compute_matches(self, image_a: Image, image_b: Image) -> list:
        """
        Compute matches between image_a and image_b.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            matches: Matches between image_a and image_b
        """
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = []

        raw_matches = matcher.knnMatch(image_a.features, image_b.features, 2)
        matches = []

        for m, n in raw_matches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * self.ratio:
                matches.append(m)
                
        #print("matches m ", matches)
        return matches
