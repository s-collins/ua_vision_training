import subprocess


if __name__ == '__main__':
    subprocess.call([
        'python',
        '/home/paperspace/models/research/object_detection/legacy/eval.py',
        '--logtostderr',
        '--pipeline_config_path=models/ssd_mobilenet_v1_coco/configuration.config',
        '--checkpoint_dir=output',
        '--eval_dir=eval'
    ])
