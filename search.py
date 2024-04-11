import numpy as np
import torch
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

from utilities.utils import *

def get_similar_images(image, model_name='clip-ViT-B-32', path2cache='cache/'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index: faiss.Index = faiss.read_index('data/images.index')
    model = SentenceTransformer(
        model_name_or_path=model_name,
        cache_folder=path2cache,
        device = device
    )
    image_embedding = model.encode(image)
    D, I = index.search(np.array([image_embedding]), 15)
    
    index2image = lambda i: sorted(pull_images('data/'))[i]
    similar_images = [Image.open(index2image(i)) for i in I[0]]
    
    # show_sample(similar_images)
    return similar_images

if __name__ == '__main__':
    image_path = '/home/muhammet/Downloads/veste-simili-cuir.jpg'
    image = Image.open(image_path)
    get_similar_images(image)