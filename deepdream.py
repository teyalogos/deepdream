"""
Modified version of Google's deep dream scripts for easier and more customized use.
"""
import numpy as np
import scipy.ndimage as nd
import PIL
from google.protobuf import text_format
import argparse
import sys
import os

import caffe

# Setup commandline arguments
parser = argparse.ArgumentParser(description="This is a modified version of Google's deep dream script for easier and faster usage.")
parser.add_argument('--model', help='Model to be used in deep dream', nargs='?', const=1)
parser.add_argument('--image', help='Path of image to be deep dreamed', nargs='?', const=1)
parser.add_argument('--is_gpu', help='Indicates whether the program uses a GPU or not (y/n)', default='n', nargs='?', const=1)
parser.add_argument('--model_path', help='Path of the model to be used in deep dream', default='', nargs='?', const=1)
parser.add_argument('--frame_folder', help='Folder where deep dream frames are saved to', default='frames/', nargs='?', const=1)
parser.add_argument('--iterations', help='Number of iterations to run deep dream for', default=10, nargs='?', const=1, type=int)
parser.add_argument('--octave_n', help='Number of octaves (scales) to run deep dream for', default=4, nargs='?', const=1, type=int)
parser.add_argument('--octave_scale', help='Scale magnitude of each octave', default=1.4, nargs='?', const=1, type=float)
parser.add_argument('--layer', help="Layer to maximize. Specify '?' to print all layers or specify 'all' to maximize all layers", default='inception_4c/output', nargs='?', const=1)
parser.add_argument('--zoom', help='How much to zoom into each image every iteration', default=0, nargs='?', const=1, type=float)
parser.add_argument('--deep_iterations', help='Number of iterations to deep dream the image', default=100, nargs='?', const=1, type=int)

args = parser.parse_args()

# some exceptions
if not args.image or not args.model:
    parser.print_help()
    sys.exit(0)

# check if program is to use the GPU
if args.is_gpu == 'y':
    caffe.set_device(0)
    caffe.set_mode_gpu()

# model specification
model_path = args.model_path # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + args.model

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))


net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def objective_L2(dst):
    dst.diff[:] = dst.data 


def make_step(net, step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)


def deepdream(net, base_img, iter_n=args.iterations, octave_n=args.octave_n, octave_scale=args.octave_scale, 
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in range(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            print(octave, i, end, vis.shape)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


def run(frame, end=args.layer, folder_start=''):
    '''Run deep dream deep_iteration times'''
    frame_i = 0
    layer_dir = folder_start + '-'.join(end.split('/'))

    for i in range(args.deep_iterations):
        try:
            frame = deepdream(net, frame, end=end)
            PIL.Image.fromarray(np.uint8(frame)).save(layer_dir + "/%04d.jpg"%frame_i)
        except FileNotFoundError:
            os.makedirs(layer_dir)
        except ValueError:
            break

        # zoom image
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        frame_i += 1
        

# load image
frame = np.float32(PIL.Image.open(args.image))

# process frames
h, w = frame.shape[:2]
s = args.zoom # scale coefficient

if args.layer == '?':
    for layer in net.blobs.keys():
        print(layer)
elif args.layer == 'all':
    # iterate and deep dream all layers
    layer_num = 0
    for layer in net.blobs.keys():
        run(frame, end=layer, folder_start=str(layer_num) + '-')
        layer_num += 1
else:
    run(frame)
