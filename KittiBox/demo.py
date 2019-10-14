#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Detects Cars in an image using KittiBox.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiBox weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
sys.path.insert(1, 'incl')
from utils import train_utils as kittibox_utils

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)



class KittiBoxDetector: 
    def __init__(self):
        self.flags = tf.app.flags
        self.FLAGS = self.flags.FLAGS
        self.flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
        self.flags.DEFINE_string('input_image_path', None,
                            'Image to apply KittiBox.')
        self.flags.DEFINE_string('output_image', None,
                            'Image to apply KittiBox.')
        self.default_run = 'KittiBox_pretrained'
        self.weights_url = ("ftp://mi.eng.cam.ac.uk/"
                    "pub/mttt2/models/KittiBox_pretrained.zip")
        # tf.app.run(self.main)

    # def main(self, args):
        tv_utils.set_gpus_to_use()
        # if self.FLAGS.input_image_path is None:
        #     logging.error("No input_image was given.")
        #     logging.info(
        #         "Usage: python demo.py --input_image data/test.png "
        #         "[--output_image output_image] [--logdir /path/to/weights] "
        #         "[--gpus GPUs_to_use] ")
        #     exit(1)

        if self.FLAGS.logdir is None:
            # Download and use weights from the MultiNet Paper
            if 'TV_DIR_RUNS' in os.environ:
                self.runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                        'KittiBox')
            else:
                self.runs_dir = '../KittiBox/RUNS/'
            # print(self.runs_dir) 
            # exit(1) 
            self.maybe_download_and_extract(self.runs_dir)
            self.logdir = os.path.join(self.runs_dir, self.default_run)
        else:
            logging.info("Using weights found in {}".format(self.FLAGS.logdir))
            self.logdir = self.FLAGS.logdir

        # Loading hyperparameters from logdir
        print("Log dir is:    ", self.logdir)
        self.hypes = tv_utils.load_hypes_from_logdir(self.logdir, base_path='hypes')

        logging.info("Hypes loaded successfully.")

        # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
        modules = tv_utils.load_modules_from_logdir(self.logdir)
        logging.info("Modules loaded successfully. Starting to build tf graph.")

        # Create tf graph and build module.
        with tf.Graph().as_default():
            # Create placeholder for input
            self.image_pl = tf.placeholder(tf.float32)
            self.image = tf.expand_dims(self.image_pl, 0)

            # build Tensorflow graph using the model from logdir
            self.prediction = core.build_inference_graph(self.hypes, modules,
                                                    image=self.image)

            logging.info("Graph build successfully.")

            # Create a session for running Ops on the Graph.
            self.sess = tf.Session()
            self.saver = tf.train.Saver()

            # Load weights from logdir
            core.load_weights(self.logdir, self.sess, self.saver)

            logging.info("Weights loaded successfully.")

    def detect(self, image):
        # image = scp.misc.imread(input_image_path + input_image)
        # print(image.shape)
        # print(hypes["image_height"])
        # print(hypes["image_width"])
        image = scp.misc.imresize(image, (384,682),interp='cubic')
        image = np.pad(image, [(0, 0), (283, 283), (0, 0)], 'constant', constant_values = [(0,0),(0,0),(0,0)])
        # print(image.shape)
        feed = {self.image_pl: image}
        
        # Run KittiBox model on image
        pred_boxes = self.prediction['pred_boxes_new']
        pred_confidences = self.prediction['pred_confidences']

        (np_pred_boxes, np_pred_confidences) = self.sess.run([pred_boxes,
                                                        pred_confidences],
                                                        feed_dict=feed)

        # Apply non-maximal suppression
        # and draw predictions on the image
        # TODO: Resize the bounding box back to normal size 
        rectangles = kittibox_utils.add_rectangles(
            self.hypes, [image], np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=1,
            min_conf=0.50, tau=self.hypes['tau'], color_acc=(0, 255, 0))

        return rectangles


    def maybe_download_and_extract(self, runs_dir):
        logdir = os.path.join(runs_dir, self.default_run)

        if os.path.exists(logdir):
            # weights are downloaded. Nothing to do
            return

        if not os.path.exists(runs_dir):
            
            os.makedirs(runs_dir)


        import zipfile
        download_name = tv_utils.download(self.weights_url, runs_dir)

        logging.info("Extracting KittiBox_pretrained.zip")

        zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

        return






    # def main(self):
    
    #     tv_utils.set_gpus_to_use()

    #     if FLAGS.input_image_path is None:
    #         logging.error("No input_image was given.")
    #         logging.info(
    #             "Usage: python demo.py --input_image data/test.png "
    #             "[--output_image output_image] [--logdir /path/to/weights] "
    #             "[--gpus GPUs_to_use] ")
    #         exit(1)

    #     if FLAGS.logdir is None:
    #         # Download and use weights from the MultiNet Paper
    #         if 'TV_DIR_RUNS' in os.environ:
    #             runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
    #                                     'KittiBox')
    #         else:
    #             runs_dir = 'RUNS'
    #         maybe_download_and_extract(runs_dir)
    #         logdir = os.path.join(runs_dir, default_run)
    #     else:
    #         logging.info("Using weights found in {}".format(FLAGS.logdir))
    #         logdir = FLAGS.logdir

    #     # Loading hyperparameters from logdir
    #     print("Log dir is:    ", logdir)
    #     hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    #     logging.info("Hypes loaded successfully.")

    #     # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    #     modules = tv_utils.load_modules_from_logdir(logdir)
    #     logging.info("Modules loaded successfully. Starting to build tf graph.")

    #     # Create tf graph and build module.
    #     with tf.Graph().as_default():
    #         # Create placeholder for input
    #         image_pl = tf.placeholder(tf.float32)
    #         image = tf.expand_dims(image_pl, 0)

    #         # build Tensorflow graph using the model from logdir
    #         prediction = core.build_inference_graph(hypes, modules,
    #                                                 image=image)

    #         logging.info("Graph build successfully.")

    #         # Create a session for running Ops on the Graph.
    #         sess = tf.Session()
    #         saver = tf.train.Saver()

    #         # Load weights from logdir
    #         core.load_weights(logdir, sess, saver)

    #         logging.info("Weights loaded successfully.")

    #     input_image_path = FLAGS.input_image_path
    #     logging.info("Starting inference using {} as input image folder".format(input_image_path))

    #     # Load and resize input image
    #     # change it to load all the images 
    #     images = os.listdir(input_image_path)
    #     for input_image in images:
    #         image = scp.misc.imread(input_image_path + input_image)
    #         # print(image.shape)
    #         # print(hypes["image_height"])
    #         # print(hypes["image_width"])

    #         image = scp.misc.imresize(image, (384,
    #                                         682),
    #                                 interp='cubic')
    #         image = np.pad(image, [(0, 0), (283, 283), (0, 0)], 'constant', constant_values = [(0,0),(0,0),(0,0)])

    #         # print(image.shape)
    #         feed = {image_pl: image}
            
    #         # Run KittiBox model on image
    #         pred_boxes = prediction['pred_boxes_new']
    #         pred_confidences = prediction['pred_confidences']

    #         (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
    #                                                         pred_confidences],
    #                                                         feed_dict=feed)

    #         # Apply non-maximal suppression
    #         # and draw predictions on the image
    #         output_image, rectangles = kittibox_utils.add_rectangles(
    #             hypes, [image], np_pred_confidences,
    #             np_pred_boxes, show_removed=False,
    #             use_stitching=True, rnn_len=1,
    #             min_conf=0.50, tau=hypes['tau'], color_acc=(0, 255, 0))

    #         # threshold = 0.5
    #         # accepted_predictions = []
    #         # # removing predictions <= threshold
    #         # for rect in rectangles:
    #         #     if rect.score >= threshold:
    #         #         accepted_predictions.append(rect)

    #         # print('')
    #         logging.info("{} Cars detected".format(len(rectangles)))

    #         # Printing coordinates of predicted rects.
    #         #for i, rect in enumerate(rectangles):
    #         #    logging.info("")
    #         #    logging.info("Coordinates of Box {}".format(i))
    #         #    logging.info("    x1: {}".format(rect.x1))
    #         #    logging.info("    x2: {}".format(rect.x2))
    #         #    logging.info("    y1: {}".format(rect.y1))
    #         #    logging.info("    y2: {}".format(rect.y2))
    #         #    logging.info("    Confidence: {}".format(rect.score))
            

    #         # save Image path TBD
    #         if FLAGS.output_image is None:
    #             output_name = input_image.split('.')[0] + '_rects.jpg'
    #         else:
    #             output_name = FLAGS.output_image + input_image.split('.')[0] + '_rects.jpg'

    #         scp.misc.imsave(output_name, output_image)

    #     logging.info("DONE!!!!!!!!!!!!!")
    
# if __name__ == '__main__':
#     tf.app.run()


