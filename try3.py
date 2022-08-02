import numpy as np
from scipy.ndimage import maximum_filter
import scipy
import matplotlib.pyplot as plt


def main():

    kernel = (plt.imread("traffic_light.png")/255)
    kernel = kernel[:, :, 0]
    kernel -= np.mean(kernel)
    print(kernel)
    img = plt.imread("berlin_000521_000019_leftImg8bit.png")
    img = img[:, :, 0]

    array = scipy.ndimage.convolve(img, kernel)

    plt.imshow(array)
    print("done")


if __name__ == '__main__':
    main()
