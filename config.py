import os

data_path = '/kaggle/input/deepfake-face-mask-dataset-dffmd'
fake_path = os.path.join(data_path, 'Fake', 'Fake')
real_path = os.path.join(data_path, 'Real', 'Real')
metadata_path = os.path.join(data_path, 'DFFD_metadata.json')

IMG_SIZE = 128
FRAMES_PER_VIDEO = 10
LIMITED_VIDEOS = 100