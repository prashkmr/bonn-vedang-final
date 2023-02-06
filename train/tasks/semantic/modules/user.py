#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
import numba

from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    print("l69")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      print("l71")
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      print("l75")
      self.model.cuda()
    
    self.stat_classes = []

  def infer(self):
    print("l78")
    # do train set
    self.infer_subset(loader=self.parser.get_train_set(),
                      to_orig_fn=self.parser.to_original)
    print("l84")

    # do valid set
    self.infer_subset(loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original)
    # do test set
    self.infer_subset(loader=self.parser.get_test_set(),
                      to_orig_fn=self.parser.to_original)

    print('Finished Infering')

    return

  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, unproj_xyz, _, unproj_remissions, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]
        # print("p_x: ", p_x.shape)
        # print("p_x: ", p_x)
        # print("p_y: ", p_y.shape)
        # print("p_y: ", p_y)
        # print("proj_range: ", proj_range.shape)
        # print("unproj_range: ", unproj_range.shape)
        # print("proj_in: ", proj_in.shape)
        # print("proj_mask: ", proj_mask.shape)

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          # proj_labels = proj_labels.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        # compute output
        proj_output = self.model(proj_in, proj_mask)
        
        # print("output: ", proj_output.clamp(min=1e-8).shape)
        # print("proj_labels: ", len(proj_labels))
        print(proj_output.shape)
        proj_argmax = proj_output[0].argmax(dim=0)

        if self.post:
          # knn postproc
          # print("post")
          unproj_argmax = self.post(proj_range,
                                    unproj_range,
                                    proj_argmax,
                                    p_x,
                                    p_y)
        else:
          # put in original pointcloud using indexes
          # print("unproj")
          unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
          torch.cuda.synchronize()

        print("Infered seq", path_seq, "scan", path_name,
              "in", time.time() - end, "sec")
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        # print("pred_np: ", pred_np.shape)
        pred_np = to_orig_fn(pred_np)

        # save scan
        # print("pred_np2: ", pred_np.shape)
        # print("pred_np2: ", np.unique(pred_np))
        
        above_gnd = np.array([50, 51, 70, 71, 80, 81])
        above_gnd_red = np.array([13, 14, 15, 16, 18, 19])
        # use numpy to keep only those classes which are above ground
        # pred_np = np.array([x if x in above_gnd else 0 for x in pred_np])
        # # @numba.njit
        # def above_ground(x: np.array):#, y: np.array):
        #     temp = np.array([44, 48, 50, 51, 70, 71, 80, 81])
        #     return np.where(np.isin(x, temp), x, -1)
        # vfunc = above_ground
        # pred_np = vfunc(pred_np)#, above_gnd)
        # print("pred_np3: ", np.unique(pred_np))
        # @numba.jit(nopython=True, parallel=True, cache=True)
        # @numba.njit
        def new_cloud_bonnetal(points, labels, above_gnd):
            lidar_data = points[:, :2]  # neglecting the z co-ordinate
            height_data = points[:, 2] #+ 1.732
            points2 = np.zeros((points.shape[0],4), dtype=np.float32) - 1
            # lidar_data -= grid_sub
            # lidar_data = lidar_data /voxel_size # multiplying by the resolution
            # lidar_data = np.floor(lidar_data)
            # lidar_data = lidar_data.astype(np.int32)
            # above_gnd = np.array([44, 48, 50, 51, 70, 71, 80, 81])
            N = lidar_data.shape[0] # Total number of points
            for i in numba.prange(N):
                # x = lidar_data[i,0]
                # y = lidar_data[i,1]
                # z = height_data[i]
                # if (0 < x < elevation_map.shape[0]) and (0 < y < elevation_map.shape[1]):
                #     if z > elevation_map[x,y] + threshold:
                #         points2[i,:] = points[i,:]
                if labels[i] in above_gnd:
                    points2[i,:] = points[i,:]
            points2 = points2[points2[:,0] != -1]
            return points2
        # print("proj_xyz: ", unproj_xyz.shape)
        # print("proj_remissions: ", unproj_remissions.unsqueeze(2).shape)
        points = torch.cat([unproj_xyz[:, :len(pred_np), :], unproj_remissions[:, :len(pred_np)].unsqueeze(2)], dim=2).cpu().numpy()
        # points = np.hstack([unproj_xyz, unproj_remissions.unsqueeze(2)])
        # print(points[0].shape)
        # print("proj labels: ", unproj_labels[0].shape)
        # import sys
        # sys.stdout.flush()
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
        unproj_labels = to_orig_fn(unproj_labels)
        # print("labels written")
        # sys.stdout.flush()
        path2 = os.path.join(self.logdir, "sequences",
                            path_seq, "true_points", path_name[:-6]+".bin")
        # cld = new_cloud_bonnetal(points[0].copy(), unproj_labels[0].cpu().numpy().copy(), above_gnd=above_gnd)
        cld = new_cloud_bonnetal(points[0].copy(), unproj_labels[0].copy(), above_gnd=above_gnd)
        print(np.unique(unproj_labels))
        print(np.unique(proj_labels))
        print(np.unique(pred_np))
        print(np.unique(self.DATA["labels"]))
        cld.tofile(path2)
        # print("true bonnetal written")
        # sys.stdout.flush()
        path2 = os.path.join(self.logdir, "sequences",
                            path_seq, "pred_points", path_name[:-6]+".bin")
        new_cloud_bonnetal(points[0].copy(), pred_np.copy(), above_gnd=above_gnd).tofile(path2)

