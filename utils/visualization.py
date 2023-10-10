import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy import ndarray
import os
from typing import Union

def plot_any(arr:list,puzzle_size:Union[tuple, list], title:ndarray= None,image_name:str = None,save_path = 'output'):
    """
    Plot a grid of images with optional titles and save the plot as an image.

    Args:
        arr (list): A list of NumPy arrays, each representing an image to be plotted.
        puzzle_size (tuple or list): A tuple or list specifying the number of puzzle pieces
                                      used during image segmentation.
        title (numpy.ndarray, optional): An optional NumPy array of titles corresponding to each image.
        image_name (str, optional): The name of the image file to save. If not provided, the plot will not be saved.
        save_path (str, optional): The directory where the image should be saved. Defaults to 'output'.

    Example:
        To plot a grid of images with titles and save the plot as 'output.png', you can call the function as follows:
        plot_any(image_list, puzzle_size=(2, 3), title=title_array, image_name='output.png', save_path='output_dir')
    """

    plt.figure(figsize = (10, 10))
    nrows, ncols = puzzle_size
    colors = list(mcolors.TABLEAU_COLORS)
    for i in range(len(arr)):
        plt.subplot(nrows,ncols,i + 1)
        plt.axis('off')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.1, hspace=0.1)
        t = title[i] if title is not None else i
        if t == 1000:
            t = ''
            color_idx = 'w'
        else:
            color_idx = colors[t] if title is not None else 'r'
        plt.title(t,x=0.5,y=0.5,color=color_idx,font=dict(weight="bold",size=28))
        plt.imshow(arr[i])
    if image_name is not None:
        plt.savefig(os.path.join(save_path,image_name),bbox_inches='tight')
    plt.close()