



try:
    from create_crop_images import main_crop_image
    import os
    import json
    import glob
    import argparse

    import numpy as np
    import pandas as pd
    from pathlib import Path
    import re
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    ### WRITE YOUR CODE HERE ###
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
    Path(r"..\cropped_images").mkdir(parents=True, exist_ok=True)
    df = pd.read_hdf("attention_results.h5")
    default_base = r"..\traffic_lights\leftImg8bit_trainvaltest\train\aachen"
    new_table = pd.DataFrame(columns=['seq', 'is_true', 'is_ignore', 'crop_path', 'original_path',
                                      'x0', 'x1', 'y0', 'y1', 'col'])

    if args.dir is None:
       args.dir = default_base
    flist = [os.path.join(root, file) for root, dirs, files in os.walk(args.dir) for file in files
             if file.endswith("_leftImg8bit.png")]
    # flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    # flist =
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        # test_find_tfl_lights(image, json_fn)
        main_crop_image(image, df.loc[df["path"] == re.search(r"([^\\]+\.png)$", image).group(0)], new_table)
        # colored = image.replace('_leftImg8bit', '_gtFine_color')
        # colored = colored.replace('leftImg8bit_trainvaltest', 'gtFine')
        # print(colored)
        #
        # image = np.array(Image.open(colored))
        # plt.imshow(image)
        # plt.show()
        # match = re.search(r"([^\\]+\.png)$", image)
        # print(match.group(0))
    new_table.to_hdf('new_table.h5', key='new_table', mode='w')
    # if len(flist):
    #    print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    # else:
    #     print("Bad configuration?? Didn't find any picture to show")
    # plt.show(block=True)


if __name__ == '__main__':
    main()
    df = pd.read_hdf("new_table.h5")
    pd.set_option('display.width', 200, 'display.max_rows', None, 'display.max_columns', 200, 'max_colwidth', 100)
    print(df)
