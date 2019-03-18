import subprocess


if __name__ == '__main__':
    subprocess.call([
        'python',
        '/home/paperspace/models/research/object_detection/legacy/eval.py',
        '--logtostderr',
        '--pipeline_config_path=output/configuration.config',
        '--checkpoint_dir=output',
        '--eval_dir=eval'
    ])
