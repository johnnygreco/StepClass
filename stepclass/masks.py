import numpy as np
import sep
from .log import logger

__all__ = ['sep_ellipse_mask']


def sep_ellipse_mask(sources, image_shape, scale=5.0):

    logger.info('building ellipse mask')

    mask = np.zeros(image_shape, dtype=bool)

    sep.mask_ellipse(mask, sources['x'], sources['y'], sources['a'],
                     sources['b'], sources['theta'], scale)

    logger.info('{:.2f}% of image masked'.format(100 * mask.sum()/mask.size))

    return mask
