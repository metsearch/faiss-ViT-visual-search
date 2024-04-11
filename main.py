import click

import gradio as gr

import torch
import faiss
from sentence_transformers import SentenceTransformer

from PIL import Image

from search import get_similar_images
from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug 
    invoked_subcommand = ctx.invoked_subcommand 
    if invoked_subcommand is None:
        logger.debug('no subcommand were called')
    else:
        logger.debug(f'{invoked_subcommand} was called')
 
@router_cmd.command()
@click.option('--model_name', help='The name of the model', type=str, default='clip-ViT-B-32')
@click.option('--index_dim', help='The dimension of index', type=int, default=512)
@click.option('--batch_size', help='Batch size', type=int, default=32)
@click.option('--path2data', help='Path to source data', type=str, default='data/')
@click.option('--path2index', help='Path to index', type=str, default='data/images.index')
@click.option('--path2cache', help='Path to cache', type=str, default='cache/')
def vectorize(model_name, index_dim, batch_size, path2data, path2index, path2cache):
    if not os.path.exists(path2cache):
        os.makedirs(path2cache)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index = faiss.IndexFlatL2(index_dim)
    
    model = SentenceTransformer(
        model_name_or_path=model_name,
        cache_folder=path2cache,
        device = device
    )
    
    image_paths = pull_images(path2data)
    image_paths = sorted(image_paths)
    
    nb_images = len(image_paths)
    logger.info(f'Number of images: {nb_images}')
    
    for cursor in range(0, nb_images, batch_size):
        sample = image_paths[cursor:cursor+batch_size]
        image_accumulator = []
        for image_path in sample:
            with Image.open(fp=image_path) as fp:
                image_copy = fp.copy()
                image_accumulator.append(image_copy)
        
        image_embeddings = model.encode(
            sentences=image_accumulator,
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        index.add(image_embeddings)
        
    faiss.write_index(index, path2index)
    logger.info(f'Index saved to {path2index}')

@router_cmd.command()
def search():
    with gr.Blocks(theme=gr.themes.Monochrome(), title='Fashion Visual Search') as app:
        query_image = gr.Image(type='pil')
        submit_button = gr.Button('Submit', scale=0)
        gallery = gr.Gallery(label='Similar images', show_label=False, elem_id='gallery', columns=[5], rows=[3], object_fit='contain', height='auto')

        submit_button.click(get_similar_images, inputs=[query_image], outputs=[gallery])
    
    # image_embedding = model.encode(image)
    # D, I = index.search(np.array([image_embedding]), 15)
    
    # index2image = lambda i: sorted(pull_images('data/'))[i]
    # similar_images = [Image.open(index2image(i)) for i in I[0]]
    app.launch()
    
if __name__ == '__main__':
    router_cmd(obj={})