from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
import io
import logging
import os
from lxml import etree
import PIL.Image
import random
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

#-------------------------------------------------------------------------------
# Command-line arguments
#-------------------------------------------------------------------------------

flags = tf.app.flags
flags.DEFINE_string('data_dir',                 '', 'Directory of PASCAL data')
flags.DEFINE_string('training_set_output_path', '', 'Path to training TFRecord')
flags.DEFINE_string('eval_set_output_path',     '', 'Path to eval TFRecord')
flags.DEFINE_string('num_examples',             '', 'Num. training examples')
flags.DEFINE_string('training_ratio',           '', 'Percentage of training examples')
flags.DEFINE_string('label_map_path',           '', 'Path to label map')
ARGS = flags.FLAGS

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------

def get_encoded_jpeg(image_path):
    """
    Returns encoded JPEG data. Throws exception if image at given path is
    not a JPEG.
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = PIL.Image.open(io.BytesIO(encoded_jpg))
    if image.format != 'JPEG':
        raise ValueError('Image is not a JPEG')
    return encoded_jpg

def dict_to_tf_example(data_dict, label_map_dict, data_dir):
    """
    Creates training example object (see tf.train.Example)

    Arguments:
        data_dict       dictionary created from a PASCAL VOC annotation file
        label_map_dict  dictionary containing class_id to class_name mappings

    Returns:
        A tf.train.Example object containing bounding box annotation data
        as well as encoded JPEG data.
    """

    # Extract information from dictionary
    image_filename = data_dict['filename']
    width = int(data_dict['size']['width'])
    height = int(data_dict['size']['height'])

    # Get JPEG data as encoded bytes
    image_path = os.path.join(data_dir, image_filename)
    encoded_jpg = get_encoded_jpeg(image_path)

    # Create array of class labels for the annotations (i.e., bounding boxes)
    # associated with this training example
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    class_ids = []
    class_names = []
    if 'object' in data_dict:
        for obj in data_dict['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            class_ids.append(label_map_dict[obj['name']])
            class_names.append(obj['name'].encode('utf8'))

    key = hashlib.sha256(encoded_jpg).hexdigest()

    # Create training example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(class_names),
        'image/object/class/label': dataset_util.int64_list_feature(class_ids)
    }))
    return example

def get_partitioned_ids(num_examples, training_ratio):
    """
    Returns two sets of ids between 1 and total number of training examples
    (inclusive).

    The ids are randomly assigned to each set such that the
    command-line argument "training_ratio" is satisfied.  Note that the
    training ratio is the number of training examples reserved for training
    divided by the total number of training examples.
    """
    num_examples = int(num_examples)
    ratio = float(training_ratio)
    full_range = range(1, num_examples + 1)
    training_ids = random.sample(full_range, int(ratio * num_examples))
    eval_ids = list(set(full_range) - set(training_ids))
    return (training_ids, eval_ids)

def write_tf_record(output_path, training_example_ids, label_map_path, data_dir):
    """
    Writes TFRecord for a set of training examples.
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in training_example_ids:
        path = os.path.join(data_dir, str(i) + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data_dict = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(data_dict, label_map_dict, data_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()   

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

def main(data_dir, training_set_output_path, eval_set_output_path, num_examples,
         training_ratio, label_map_path):
    (training_set, eval_set) = get_partitioned_ids()
    write_tf_record(training_set_output_path, training_set, label_map_path, data_dir)
    write_tf_record(eval_set_output_path, eval_set, label_map_path, data_dir)


#def main(_):
    #(training_set, eval_set) = get_partitioned_ids(num_examples, training_ratio)
    #write_tf_record(ARGS.training_set_output_path, training_set)
    #write_tf_record(ARGS.eval_set_output_path, eval_set)

#if __name__ == '__main__':
    #random.seed()
    #tf.app.run()
