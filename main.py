import numpy

try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt
    from scipy.ndimage import maximum_filter
    import scipy
    import cv2
except ImportError:
    print("Need to fix the installation")
    raise


def only_red_points(arr, original_img):
    return arr


def find_light_points(convolve_image, original_img=None) -> np.array:
    # only 2 dim image
    ret, im2 = cv2.threshold(convolve_image, 190, 255, cv2.THRESH_BINARY)
    im3 = cv2.dilate(im2, None, iterations=2)
    print(type(im3))

    img = Image.fromarray(im3, 'RGB')
    img.save('my.png')
    img.show()

    print(len(np.argwhere(im3 != 0.)))
    # return only_red_points(np.argwhere(im3 == 255), original_img)


def convolution(kernel_path, img: Image):
    """
    The function calculate the convolution of the kernel with the image.
    :param kernel_path: the path of the kernel
    :param img: the image to do the convolution with.
    :return:
    """
    kernel = (plt.imread(kernel_path) / 255)
    kernel = kernel[:, :, 0]
    kernel -= np.mean(kernel)

    img = img[:, :, 0]  # filter the red color for the image
    img = img/255
    array = scipy.ndimage.convolve(img, kernel)
    plt.imshow(array)
    plt.show()

    return array

    # plt.savefig("my_"+image_path)


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    ### WRITE YOUR CODE HERE ###
    # print(convolve_all_images(r"C:\Users\97252\bootcamp\projects\mobileye\berlin" +
    #                           "\*leftImg8bit.png", "traffic_light.png"))

    convolution_img = convolution("traffic_light.png", c_image)

    index_list = find_light_points(convolution_img)
    print(index_list)


    ### USE HELPER FUNCTIONS ###
    return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    plt.figure(56)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(image)
    plt.figure(57)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(image)

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = r"C:\Users\97252\bootcamp\projects\mobileye\berlin1"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        print(image)
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()


