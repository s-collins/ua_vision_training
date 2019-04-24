from utilities import create_tf_record


if __name__ == '__main__':

	# Create TFRecords for training and evaluation sets
	options = {
		'data_dir': 'data',
		'tfrecord_dir': 'data/tfrecords',
		'label_map_path': 'models/label_map.pbtxt',
		'num_examples': 300,
		'k': 5
	}
	create_tf_record.cross_validation(**options)
