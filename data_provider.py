from functools import partial
'''
from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.builder import rescale_images_to_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
'''
from pathlib import Path

#import joblib
#import menpo.feature
#import menpo.image
#import menpo.io as mio
import numpy as np
import tensorflow as tf
import cv2
# import detect
import utils
_build_ref_ = True
slim = tf.contrib.slim
import os

FLAGS = tf.app.flags.FLAGS
_TRAIN_IMAGES_ = '/home/dhruv/Projects/PersonalGit/Data/output/images/'
_LABEL_PATH_ = '/home/dhruv/Projects/PersonalGit/Data/output/pts/'
_MEAN_PATH_ = '/home/dhruv/Projects/PersonalGit/Data/output/inits/'
_TRAIN_IMAGES_ = '/home/dhruv/Projects/Datasets/Datasetdir/Facelandmark/custom/menpoout/images/'
_LABEL_PATH_ = '/home/dhruv/Projects/Datasets/Datasetdir/Facelandmark/custom/menpoout/pts/'
_MEAN_PATH_ = '/home/dhruv/Projects/Datasets/Datasetdir/Facelandmark/custom/menpoout/inits/'
_VAL_IMAGES_ ='/home/dhruv/Projects/Datasets/Datasetdir/Facelandmark/custom/menpoout/val'
_PATCHES_ = 90
_PAD_WIDTH_ = 199

def pad_image(image, init_pts, mean_pts, shape):

    canvas = np.zeros(shape + [3], np.uint8)
    im_shape = image.shape
    index = 0
    if im_shape[0] <= im_shape[1]:
        index = 1

    resize_factor = shape[1] / float(im_shape[index])
    new_shape = [0, 0]
    new_shape[index] = shape[1]
    new_shape[abs(index-1)] = int(resize_factor * im_shape[abs(index - 1)])
    image = cv2.resize(image, (new_shape[1], new_shape[0]), 0)
    canvas[0: new_shape[0], 0:new_shape[1]] = image

    init_pts[:, abs(index-1)] = init_pts[:, abs(index-1)] * new_shape[index]
    init_pts[:, index] =  init_pts[:, index] * new_shape[abs(index-1)]


    mean_pts[:, abs(index-1)] = mean_pts[:, abs(index-1)] * new_shape[index] / im_shape[index]
    mean_pts[:, index] =  mean_pts[:, index] * new_shape[abs(index-1)]/ im_shape[abs(index-1)]
    #canvas = canvas.astype(np.float32) / 255.0
    #canvas -= 0.5
    #canvas *= 2.0
    return canvas, init_pts, mean_pts

def get_classification_image(path):
    image = cv2.imread(path.decode())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def get_landmarks_and_mean(name):
    lmpts_name = _LABEL_PATH_ + name + '.pts'
    lmpts = get_pts(lmpts_name)
    mean_pts = get_pts(_MEAN_PATH_ + name + '.pts')
    #print(mean_pts)
    #input('meanprint')
    #mean_pts = get_pts(_MEAN_PATH_
    return lmpts, mean_pts


def get_pts(filename):
    with open(filename) as f:
        data= f.read().split()
        data = np.asarray(data, np.float32)
        data = np.reshape(data, (_PATCHES_, 2))
    return data


def align_landmark_image(image, lmpts, mean_pts):
    shape2 = image.shape
    lm_image, lmpts, init_pts = pad_image(image, lmpts, mean_pts, shape=[_PAD_WIDTH_, _PAD_WIDTH_])
    lmpts[:,0] = lmpts[:, 0] /shape2[1]
    lmpts[:,1] = lmpts[:, 1] /shape2[0]


    lm_image = lm_image.astype(np.float32)
    lm_image = lm_image / 255.0
    '''
    lm_image -= 0.5
    lm_image *= 2.0
    '''
    return lm_image, lmpts, init_pts

def get_image_tags_points(path):
    name = os.path.split(path.decode())[1]
    name = os.path.splitext(name)[0]

    image = get_classification_image(path)
    #GT_data = get_tags(name)

    lmpts, mean_pts = get_landmarks_and_mean(name)

    lm_image, lmpts, init_pts = align_landmark_image(image.copy(), lmpts.copy(), mean_pts.copy())

    return lm_image, lmpts, init_pts

def decode_img_lmpts(image1):
    image, lmpts, init_pts= tf.py_func(get_image_tags_points, [image1], (tf.float32, tf.float32, tf.float32))
    #image = inception_preprocessing.preprocess_image(image, _IMAGE_HEIGHT_, _IMAGE_WIDTH_, is_training=False)
    image.set_shape([_PAD_WIDTH_, _PAD_WIDTH_, 3])
    lmpts.set_shape([_PATCHES_, 2])
    init_pts.set_shape([_PATCHES_, 2])

    return image, lmpts, init_pts


def build_reference_shape(paths, diagonal=200):
    """Builds the reference shape.

    Args:
      paths: paths that contain the ground truth landmark files.
      diagonal: the diagonal of the reference shape in pixels.
    Returns:
      the reference shape.
    """
    landmarks = []
    for path in paths:
        path = Path(path).parent.as_posix()
        landmarks += [
            group.lms
            for group in mio.import_landmark_files(path, verbose=True)
            if group.lms.n_points == _PATCHES_
        ]

    return compute_reference_shape(landmarks,
                                   diagonal=diagonal).points.astype(np.float32)

def get_bounding_box(shape):
    min_xy = tf.reduce_min(shape, 0)
    max_xy = tf.reduce_max(shape, 0)

    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]

    return tf.stack([[min_x, min_y], [max_x, min_y],
                    [max_x, max_y], [min_x, max_y]])

def align_reference_shape_w_image(reference_shape, bounding_box, im):
    reference_shape_bb = get_bounding_box(reference_shape)
    
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bounding_box) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bounding_box, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_images(im, new_size), align_mean_shape / ratio, ratio

def align_reference_shape(reference_shape, bounding_box):
    '''Aligns the reference shape to a bounding box.
    
    Args:
      reference_shape: A `Tensor` of dimensions [num_landmarks, 2].
      bounding_box: A `Tensor` of dimensions [4, 2].
    Returns:
      the similarity aligned shape to the bounding box.
    '''

    reference_shape_bb = get_bounding_box(reference_shape)

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bounding_box) / norm(reference_shape_bb)

    return tf.add(
        (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio,
        tf.reduce_mean(bounding_box, 0),
        name='initial_shape')


def random_shape(gts, reference_shape, pca_model):
    """Generates a new shape estimate given the ground truth shape.

    Args:
      gts: a numpy array [num_landmarks, 2]
      reference_shape: a Tensor of dimensions [num_landmarks, 2]
      pca_model: A PCAModel that generates shapes.
    Returns:
      The aligned shape, as a Tensor [num_landmarks, 2].
    """

    def synthesize(lms):
        return detect.synthesize_detection(pca_model, menpo.shape.PointCloud(
            lms).bounding_box()).points.astype(np.float32)

    bb, = tf.py_func(synthesize, [gts], [tf.float32])
    shape = align_reference_shape(reference_shape, bb)
    shape.set_shape(reference_shape.get_shape())

    return shape



def distort_color(image, thread_id=0, stddev=0.05, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        
        image += tf.random_normal(
                tf.shape(image),
                stddev=stddev,
                dtype=tf.float32,
                seed=42,
                name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

def grey_to_rgb(im):
    """Converts menpo Image to rgb if greyscale
    Args:
      im: menpo Image with 1 or 3 channels.
    Returns:
      Converted menpo `Image'.
    """
    assert im.n_channels in [1, 3]

    if im.n_channels == 3:
        return im

    im.pixels = np.vstack([im.pixels] * 3)
    return im

def load_image(path, reference_shape, is_training=False, group='PTS',
               mirror_image=False):
    """Load an annotated image.
    In the directory of the provided image file, there
    should exist a landmark file (.pts) with the same
    basename as the image file.
    Args:
      path: a path containing an image file.
      reference_shape: a numpy array [num_landmarks, 2]
      is_training: whether in training mode or not.
      group: landmark group containing the grounth truth landmarks.
      mirror_image: flips horizontally the image's pixels and landmarks.
    Returns:
      pixels: a numpy array [width, height, 3].
      estimate: an initial estimate a numpy array [68, 2].
      gt_truth: the ground truth landmarks, a numpy array [68, 2].
    """
    im = mio.import_image(path.decode("utf-8"))
    bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
    if 'set' not in str(bb_root):
        bb_root = im.path.parent.relative_to(im.path.parent.parent)

    im.landmarks['bb'] = mio.import_landmark_file(str(Path('bbs') / bb_root / (
        im.path.stem + '.pts')))

    im = im.crop_to_landmarks_proportion(0.3, group='bb')
    reference_shape = PointCloud(reference_shape)

    bb = im.landmarks['bb'].lms.bounding_box()

    im.landmarks['__initial'] = noisy_shape_from_bounding_box(reference_shape,
                                                              bb)
    im = im.rescale_to_pointcloud(reference_shape, group='__initial')

    if mirror_image:
        im = utils.mirror_image(im)

    lms = im.landmarks[group].lms
    initial = im.landmarks['__initial'].lms

    # if the image is greyscale then convert to rgb.
    pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)

    gt_truth = lms.points.astype(np.float32)
    init_shape = initial.points.astype(np.float32)
    return pixels.astype(np.float32).copy(), gt_truth, init_shape

def read_images(paths,
                 batch_size=32,
                 is_training=False,
                 num_landmarks=_PATCHES_,
                 mirror_image=False):
    """Reads the files off the disk and produces batches.
    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
      num_landmarks: the number of landmarks in the training images.
      mirror_image: mirrors the image and landmarks horizontally.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """
    global  _build_ref_
    if _build_ref_:
        reference_shape = PointCloud(build_reference_shape(paths))
        mio.export_pickle(reference_shape.points, 'reference_shape.pkl', overwrite=True)
        _build_ref_ = False
    reference_shape = tf.constant(mio.import_pickle('reference_shape.pkl', encoding='latin1'))
    files = tf.concat(axis=0, values=[list(map(str, sorted(Path(d).parent.glob(Path(d).name))))
                          for d in paths])

    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=is_training,
                                                    capacity=1000)

    filename = filename_queue.dequeue()

    image, lms, lms_init = tf.py_func(
        partial(load_image, is_training=is_training,
                mirror_image=mirror_image),
        [filename, reference_shape], # input arguments
        [tf.float32, tf.float32, tf.float32], # output types
        name='load_image'
    )

    # The image has always 3 channels.
    image.set_shape([None, None, 3])

    if is_training:
        image = distort_color(image)

    lms = tf.reshape(lms, [num_landmarks, 2])
    lms_init = tf.reshape(lms_init, [num_landmarks, 2])

    images, lms, inits, shapes = tf.train.batch(
                                    [image, lms, lms_init, tf.shape(image)],
                                    batch_size=batch_size,
                                    num_threads=4 if is_training else 1,
                                    capacity=1000,
                                    enqueue_many=False,
                                    dynamic_pad=True)

    return images, lms, inits, shapes

def batch_inputs(paths,
                 batch_size=32,
                 is_training=False,
                 num_channels=3, num_landmarks=_PATCHES_):
    """Reads the files off the disk and produces batches.

    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """
    global  _build_ref_
    if _build_ref_:
        reference_shape = PointCloud(build_reference_shape(paths))
        mio.export_pickle(reference_shape.points, 'reference_shape.pkl', overwrite=True)
        _build_ref_ = False

    reference_shape = tf.constant(mio.import_pickle('reference_shape.pkl', encoding='latin1') * .8)

    files = tf.concat(axis=0, values=[tf.matching_files(d) for d in paths])

    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=is_training,
                                                    capacity=1000)

    image, lms, lms_init = tf.py_func(
        partial(load_image, is_training=is_training),
        [filename_queue.dequeue(), reference_shape], # input arguments
        [tf.float32, tf.float32, tf.float32], # output types
        name='load_image'
    )

    # The image has always 3 channels.
    image.set_shape([None, None, 3])

    if is_training:
        image = distort_color(image)

    lms = tf.reshape(lms, [num_landmarks, 2])
    lms_init = tf.reshape(lms_init, [num_landmarks, 2])

    images, lms, inits = tf.train.batch([image, lms, lms_init],
                                        batch_size=batch_size,
                                        num_threads=4,
                                        capacity=1000,
                                        enqueue_many=False,
                                        dynamic_pad=True)

    return images, lms, inits

def super_batch_inputs(paths,
                 batch_size=32,
                 is_training=False,
                 num_channels=3, num_landmarks=68):
    """Reads the files off the disk and produces batches.

    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """
    if is_training:
        data_path = _TRAIN_IMAGES_
    else:
        data_path = _VAL_IMAGES_

    images = os.listdir(data_path)
    images_full = []
    for image in images:
        images_full.append(os.path.join(data_path, image))

    dataset = tf.data.Dataset.from_tensor_slices((images_full))

    if is_training:
        dataset = dataset.shuffle(buffer_size=25000).repeat()
    dataset = dataset.map(decode_img_lmpts)
    dataset = dataset.batch(batch_size)



    iterator = dataset.make_one_shot_iterator()
    images, lms, inits = iterator.get_next()

    return images, lms, inits