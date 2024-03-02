import cv2 as cv
import numpy as np
import math
import pickle
from src.images import Image
from src.images import camera
from src.matching import PairMatch, match
from src.rendering import bundle_adjuster

class camera_Estimator:
          
    
   
  def create_cameras(self, pair_matches):
      focals=[]
      for pair_match in pair_matches:
          focals.append(self.estimate_focal_from_homography(pair_match.H))
      focal = np.median(focals) 
      print("Focal camara", focal)
      
      for camera in self._all_cameras(pair_matches):
        camera.focal = focal
      return camera
      
  def max_span_tree_order_2(self, matches):
      '''
      Finds a maximum spanning tree from all matches, with most connected edge as start point
      '''
      connected_nodes = set()
      all_cameras = self._all_cameras(matches)
     
      sorted_all_cameras = sorted(all_cameras, key=lambda image:image.path)
      #[print(c.image) for c in sorted_all_cameras]
      sorted_edges = sorted(matches, key=lambda matches: len(matches.matches), reverse=True)
      [print(f'{e.image_a.image} - {e.image_b.image}: {len(e.matches)}') for e in sorted_edges]
      best_edge = sorted_edges.pop(0)

      # if (sorted_all_cameras.index(best_edge.cam_from) > sorted_all_cameras.index(best_edge.cam_to)):
      #   print("edge swapped")
      #   self._reverse_match(best_edge)
      # else:
      #   self._normalise_match_H(best_edge)

      #print(f'Best edge: {best_edge.image_a.path} - {best_edge.image_b.path}: {len(best_edge.matches)}')
      print(f'Best edge H: {best_edge.H}')
      
      print("order matches R")
      print(best_edge.image_a.focal)
      
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

  def set_bundle(self, matches, add_order):
      
      ba = bundle_adjuster.BundleAdjuster()

      other_matches = set(matches) - set(add_order)
      identity_cam = add_order[0].image_a
      identity_cam.R, identity_cam.K = np.identity(3),np.identity(3)
      identity_cam.ppx, identity_cam.ppy = 0, 0
      identity_cam = add_order[0].image_b
      identity_cam.R, identity_cam.K = np.identity(3),np.identity(3)
      identity_cam.ppx, identity_cam.ppy = 0, 0

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
      all_cameras = self._all_cameras(matches.matches)
      return all_cameras

  def _all_cameras(self, matches):
      all_cameras = set()

      for match in matches:
        all_cameras.add(match.image_a)
        all_cameras.add(match.image_b)

      return all_cameras

  def _normalise_match_H(self, match):
      return np.multiply(match.H, 1 / match.H[2][2])

  def _reverse_match(self, match):
      match.image_a, match.image_b = match.image_b, match.image_a
      match.H = np.linalg.pinv(match.H)
      for inlier in match.inliers:
        inlier[0], inlier[1] = inlier[1], inlier[0]
      return self._normalise_match_H(match)

  """ 
  def initial_bundle_adjustment(matches, add_order):
      other_matches = set(matches) - set(add_order)
      identity_cam = add_order[0].image_a
      identity_cam.R = np.identity(3)
      identity_cam.ppx, identity_cam.ppy = 0, 0

      for match in add_order:
          match.image_b.R = (match.image_a.R.T @ (np.linalg.pinv(match.image_a.K) @ match.H @ match.image_b.K)).T
          match.image_b.ppx, match.image_b.ppy = 0, 0
          ba.add(match)
          added_cams = ba.added_cameras()
          to_add = set()
          for other_match in other_matches:
              # If both cameras already added, add the match to BA
              if (other_match.image_A in added_cams and other_match.image_b in added_cams):
                to_add.add(other_match)
          for match in to_add:
              # self._reverse_match(match)
              ba.add(match)
              other_matches.remove(match)
      
      all_cameras = None
      ba.run()
      all_cameras = all_cameras()
      return all_cameras


  def all_cameras(matches):
      all_cameras = []

      for match in matches:
        all_cameras.add(match.image_a)
        all_cameras.add(match.image_b)

      return all_cameras
  """  


  def estimate_focal_from_homography(self, H):
      h = H

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