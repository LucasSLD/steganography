import numpy as np
from math import log
import matplotlib.pyplot as plt
from torchvision import transforms

to_tensor = transforms.ToTensor()

def plot_np_array(img_numpy_array, title : str = None, colorbar=False,figsize=None):
    if figsize is not None: plt.figure(figsize=figsize)
    if title is not None: plt.title(title)
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
    plt.imshow(img_numpy_array)
    if colorbar: plt.colorbar()
    plt.show()

def plot_tensor(tensor, title : str = None, colorbar=False,figsize=None):
    if len(tensor.shape) == 4:
        array = tensor[0].permute(1,2,0).cpu().numpy()
    elif len(tensor.shape) == 3:
        array = tensor.permute(1,2,0).cpu().numpy()
    else:
        print("ShapeError: the input array should be 3D or 4D")
        return
    plot_np_array(array, title, colorbar, figsize)

def read_pgm(file_name):
    with open(file_name, 'rb') as f:
        # Skip header
        f.readline()
        dimensions = f.readline().split()
        width, height = int(dimensions[0]), int(dimensions[1])
        max_val = int(f.readline())

        # Read the image data
        img = np.fromfile(f, dtype=np.uint8, count=width * height).reshape((height, width))
    return img

def ternary_entropy(p: float) -> float:
    """
    Ternary entropy
    Args:
        p (float): half insertion rate, probability p of adding 1 and proba p of substracting 1

    Returns:
        float: ternary entropy for given p
    """
    return -2*p*log(p,2) - (1-2*p)*log(1-2*p,2)

def pgm_to_tensor(file: str):
    precover = read_pgm(file)
    return to_tensor(precover).repeat(1,3,1,1).cuda()

def network_output_to_numpy(net_out) -> np.ndarray:
    """Converts output of ImageCompressor networks to numpy arrays

    Args:
        net_out (Tensor): tensor of an image produced by the ImageCompressor network (or one of its variations). It has the shape (1,channels,height,width)

    Returns:
        np.ndarray: numpy array of an image with shape (height,width,channels) and values between 0 and 1
    """
    assert len(net_out.shape) == 3 or len(net_out.shape) == 4
    if len(net_out.shape) == 4: net_out = net_out[0] # the first axis is useless when there are 4 dimensions
    to_pil = transforms.ToPILImage()
    img = np.array(to_pil(net_out.cpu()))
    return img

def diff_cover_stego(cover_t, stego_t):
    """
    Args:
        cover_t (Tensor): cover tensor of an image (output of ImageCompressor model)
        stego_t (Tensor): stego tensor of an image (output of ImageCompressor model)

    Returns:
        np.ndarray: difference between cover image and stego image
    """
    cover_np = network_output_to_numpy(cover_t).astype(int)
    stego_np  = network_output_to_numpy(stego_t).astype(int)
    return cover_np - stego_np