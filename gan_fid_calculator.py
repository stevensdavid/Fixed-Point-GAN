#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
from argparse import ArgumentParser
import sys

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="PCam", choices=["CelebA", "PCam"])
parser.add_argument("--image_format", type=str, default="jpg", choices=["jpg", "png"])
parser.add_argument("--stats_path", type=str)
config = parser.parse_args()

# Paths
if config.dataset == "CelebA":
    image_path = 'celeba/results' # set path to some generated images
    stats_path = config.stats_path
elif config.dataset == "PCam":
    image_path = 'pcam/results' # set path to some generated images
    stats_path = stats_path # training set statistics
else:
  print("Unsupported dataset")
  sys.exit(1)
inception_path = fid.check_or_download_inception(None) # download inception network

# loads all images into memory (this might require a lot of RAM!)
image_list = glob.glob(os.path.join(image_path, f"*.{config.image_format}"))
# images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

# load precalculated training set statistics
with np.load(stats_path) as f:
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics_from_files(image_list, sess, batch_size=100)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)
