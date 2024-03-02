import cv2 as cv
import numpy as np
import pickle
from src.images import Image
from src.images import camera
from src.matching import PairMatch, match
from src.rendering import bundle_adjuster

class camera_Estimator:
    def __init__(self, matches):
        self._matches = matches
    
    def estimate(self):
        #for m in self._matches:
            #print(f'Match (unordered) {m.cam_from.image.filename} and {m.cam_to.image.filename}: {len(m.inliers)}')
            # for match in matcher.matches:
            # result = warp_two_images(m.cam_to.image.image, m.cam_from.image.image, m.H)
            # cv.imshow('Result', result)
            # cv.waitKey(0)
        
        for match in self._matches:
          print("Elemento R AF")
          print("valor " , match.R)
          print("Elemento K AF")
          print("valor ", match.K)
       
        self._estimate_focal()
        """
        for match in self._matches:
          print("Elemento R EF")
          print("valor " , match.image_a.R)
          print("Elemento K EF")
          print("valor ", match.image_a.K)
          """
        add_order = self.max_span_tree_order_2()
        """
        for match in self._matches:
          print("Elemento R Addorder")
          print("valor " , match.image_a.R)
          print("Elemento K Addordr")
          print("valor ", match.image_a.K)
        """
        # Display order
        #print(f'Add order:')
        #for (i,m) in enumerate(add_order):
            #print(f'  {i} => Match {m.cam_from.image.filename} and {m.cam_to.image.filename}: {len(m.inliers)}')
        # return

        self.set_bundle(add_order)

        return self._all_cameras()

    def _estimate_focal(self):
        focals=[]
        for match in self._matches:
            focals.append(match.estimate_focal_from_homography())
        focal = np.median(focals) 
        print("Focal camara", focal)
        
        for camera in self._all_cameras():
            camera.focal = focal
        return camera
    
    def max_span_tree_order_2(self):
      '''
      Finds a maximum spanning tree from all matches, with most connected edge as start point
      '''
      connected_nodes = set()
      all_cameras = self._all_cameras()
     
      sorted_all_cameras = sorted(all_cameras, key=lambda image:image.path)
      #[print(c.image) for c in sorted_all_cameras]
      sorted_edges = sorted(self._matches, key=lambda matches: len(matches.matches), reverse=True)
      #[print(f'{e.image_a.image} - {e.image_b.image}: {len(e.matches)}') for e in sorted_edges]
      best_edge = sorted_edges.pop(0)

      # if (sorted_all_cameras.index(best_edge.cam_from) > sorted_all_cameras.index(best_edge.cam_to)):
      #   print("edge swapped")
      #   self._reverse_match(best_edge)
      # else:
      #   self._normalise_match_H(best_edge)

      #print(f'Best edge: {best_edge.image_a.path} - {best_edge.image_b.path}: {len(best_edge.matches)}')
      print(f'Best edge H: {best_edge.H}')
      
      #print("order matches R")
      #print(best_edge.image_a.focal)
      
      add_order = [best_edge]
      connected_nodes.add(best_edge.image_a)
      connected_nodes.add(best_edge.image_b)

      while (len(connected_nodes) < len(sorted_all_cameras)):
        for (i, match) in enumerate(sorted_edges):
          if (match.image_a in connected_nodes):
            # Add node as is
            edge = sorted_edges.pop(i)
            edge = self._normalise_match_H(edge)
            add_order.append(edge)
            connected_nodes.add(edge.image_a)
            connected_nodes.add(edge.image_b)
            break
          elif (match.image_b in connected_nodes):
            # Reverse node and add
            edge = sorted_edges.pop(i)

            edge = self._reverse_match(edge)

            add_order.append(edge)
            connected_nodes.add(edge.image_a)
            connected_nodes.add(edge.image_b)
            break
      
      return add_order

    def set_bundle(self, add_order):
      
      ba = bundle_adjuster.BundleAdjuster()

      other_matches = set(self._matches) - set(add_order)
      identity_cam = add_order[0].image_a
      identity_cam.R = np.identity(3)
      identity_cam.ppx, identity_cam.ppy = 0, 0
      """
      identity_cam = add_order[0].image_b
      identity_cam.R, identity_cam.K = np.identity(3),np.identity(3)
      identity_cam.ppx, identity_cam.ppy = 0, 0
      """

      for match in add_order:
          print(f'match.cam_from.R: {match.image_a.R}')
          print(f'match.cam_from.K: {match.image_a.K}')
          print(f'match.H: {match.H}')
          print(f'match.cam_to.K: {match.image_b.K}')
          match.image_b.R = (match.image_a.R.T @ (np.linalg.pinv(match.image_a.K) @ match.H @ match.image_b.K)).T
          match.image_b.ppx, match.image_b.ppy = 0, 0
          ba.add(match)
          added_cams = ba.added_cameras()
          to_add = set()
          for other_match in other_matches:
              # If both cameras already added, add the match to BA
              if (other_match.image_a in added_cams and other_match.image_b in added_cams):
                to_add.add(other_match)
          for match in to_add:
              # self._reverse_match(match)
              ba.add(match)
              other_matches.remove(match)
      all_cameras = None
      ba.run()
      all_cameras = self._all_cameras()
      return all_cameras

    def _all_cameras(self):
      all_cameras = set()

      for match in self._matches:
        all_cameras.add(match.image_a)
        all_cameras.add(match.image_b)

      return all_cameras