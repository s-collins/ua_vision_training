import subprocess

if __name__ == '__main__':
	subprocess.call([
		'python',
		'/home/paperspace/models/research/object_detection/export_inference_graph.py',
		'--pipeline_config_path=output/pipeline.config',
		'--trained_checkpoint_prefix=output/model.ckpt-5000',
		'--output_directory=output'
	])
