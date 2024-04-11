import os
from glob import glob

import matplotlib.pyplot as plt

from .log import logger

def pull_images(path2data, extension='jpg'):
    images_paths = []
    path2images = os.path.join(path2data, 'images')
    for dir in os.listdir(path2images):
        path2subdir = os.path.join(path2images, dir)
        images_paths.extend(glob(os.path.join(path2subdir, f'*.{extension}')))
    
    return images_paths

def show_sample(images_paths):
    if len(images_paths) > 5:
        images_paths = images_paths[:5]
    
    _, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i in range(5):
        axes[i].imshow(images_paths[i])
        axes[i].axis('off')
    
    plt.show()

if __name__ == '__main__':
    logger.info('Testing utils...')
    paths_ = pull_images('data/')
    print(len(paths_))