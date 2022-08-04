from PIL import Image

import numpy as np
from scipy.ndimage import maximum_filter
import scipy
import matplotlib.pyplot as plt


def main():
    img = plt.imread("berlin_000017_000019_leftImg8bit.png")
    plt.imshow(img)
    plt.show()
    print("done")


if __name__ == '__main__':
    main()
