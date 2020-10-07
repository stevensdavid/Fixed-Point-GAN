import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
import h5py
from PIL import Image
from argparse import ArgumentParser
import sys

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="PCam", choices=["CelebA", "PCam"])
parser.add_argument("--image_format", type=str, default="jpg", choices=["jpg", "png"])
config = parser.parse_args()

########
# PATHS
########
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)

# Both datasets will, after loaded, be in an RGB format between 0 and 255
# (based on inspection of files). They will be converted to float.
if config.dataset == "CelebA":
    data_path = "data/celeba/images"

    # list of paths
    image_list = glob.glob(os.path.join(data_path, f'*.{config.image_format}'))
    # print(imread(str(image_list[10])).astype(np.float32))
    # exit(0)
    # images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    output_path = 'fid_stats_celeba.npz' # path for where to store the statistics
elif config.dataset == "PCam":
    data_path = 'data/pcam' # set path to training set images

    # array-like object of images stored in h5py format
    image_list = h5py.File(os.path.join(data_path, 'test_x.h5'), 'r', swmr=True)['x']
    # print(image_list[10])
    # exit(0)
    # images = np.array([files[index, ...].astype(np.float32) for index in range(len(files))])
    output_path = 'fid_stats_pcam.npz' # path for where to store the statistics
else:
  print("Unsupported dataset")
  sys.exit(1)
print("%d images found" % len(image_list)) # These are not fully read into memory though

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if config.dataset == "CelebA":
      mu, sigma = fid.calculate_activation_statistics_from_files(image_list, sess, batch_size=100)
    elif config.dataset == "PCam":
      mu, sigma = fid.calculate_activation_statistics(image_list, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")