from tifffile import TiffFile
from czifile import CziFile as czi_read
import numpy as np


def read_tif(path):
    # TiffFile (not imread) so the file handle is closed deterministically; imread
    # returns a bare ndarray, which is not a context manager.
    with TiffFile(path) as tif:
        img = tif.asarray()
    return img

def read_czi(path):
    # max_workers=1 forces serial subblock decoding. These CZIs are mosaics with a
    # multi-resolution pyramid; with czifile's default threaded decode the overlapping
    # full-res and upsampled low-res pyramid subblocks race, so a channel comes out
    # sharp on one read and blocky (pixelated) on the next — non-deterministic pixels.
    # Serial decoding is deterministic and always yields the full-resolution result.
    with czi_read(path) as czi:
        img = czi.asarray(max_workers=1)
    return img

def read_npy(path):
    img = np.load(path)
    return img
