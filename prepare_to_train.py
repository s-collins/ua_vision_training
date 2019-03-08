from utilities import create_tf_record


if __name__ == '__main__':

	# Create TFRecords for training and evaluation sets
	options = {
		'data_dir': 'data',
		'training_set_output_path': 'data/training_set.tfrecord',
		'eval_set_output_path': 'data/eval_set.tfrecord',
		'num_examples': 300,
		'training_ratio': 1,
		'label_map_path': 'models/label_map.pbtxt'
	}
	create_tf_record.main(**options)
