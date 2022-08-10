import re
import consts
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2, y2)
    __and__ = intersection

    def __init__(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


def find_orange_rectangle_boundaries(picture, candidate_x, candidate_y, up_boundary_y, down_boundary_y,
                                     left_boundary_x, right_boundary_x, orange_color):
    """
    The function find the boundaries of the orange rectangle.
    :param picture: The picture of the color image.
    :param candidate_x: The x of the point that we check.
    :param candidate_y: The y of the point that we check.
    :param up_boundary_y: The y up boundary point of the orange rectangle.
    :param down_boundary_y: The y down boundary point of the orange rectangle.
    :param left_boundary_x: The x left boundary point of the orange rectangle.
    :param right_boundary_x: The right x boundary point of the orange rectangle.
    :param orange_color: The values of the orange color on the image.
    :return: The end values of the up_boundary_y, down_boundary_y, left_boundary_x, right_boundary_x.
    """
    while up_boundary_y > 0 and (picture[up_boundary_y][candidate_x] == orange_color).all():
        up_boundary_y -= 1
    while down_boundary_y < picture.shape[0] and (picture[down_boundary_y][candidate_x] == orange_color).all():
        down_boundary_y += 1
    while left_boundary_x > 0 and (picture[candidate_y][left_boundary_x] == orange_color).all():
        left_boundary_x -= 1
    while right_boundary_x < picture.shape[1] and (picture[candidate_y][right_boundary_x] == orange_color).all():
        right_boundary_x += 1
    return up_boundary_y, down_boundary_y, left_boundary_x, right_boundary_x


def check_rectangle_in_color_image(candidate_x: int, candidate_y: int, top_left_x: float, top_left_y: float,
                                   bottom_right_x: float, bottom_right_y: float, image_path: str) -> int:
    """
    The function get the x,y candidate for traffic light and check with the color image if it is really traffic light.
    :param candidate_x: The x of the point that we check.
    :param candidate_y: The y of the point that we check.
    :param top_left_x: The top left x point of the rectangle of the traffic light.
    :param top_left_y: The top left y point of the rectangle of the traffic light.
    :param bottom_right_x: The bottom right x point of the rectangle of the traffic light.
    :param bottom_right_y: The bottom right y point of the rectangle of the traffic light.
    :param image_path: The path of the color image.
    :return: 0 if not traffic light, 1 if it is, and 2 if ignore.
    """
    orange_color = np.array([0.98039216, 0.6666667, 0.11764706, 1.], dtype='float32')
    candidate_y = int(candidate_y)
    candidate_x = int(candidate_x)
    up_boundary_y = candidate_y
    down_boundary_y = candidate_y

    left_boundary_x = candidate_x
    right_boundary_x = candidate_x
    colored_path = image_path.replace(consts.ORIGINAL_IMAGE_SUFFIX, consts.COLORED_IMAGE_SUFFIX)
    colored_path = colored_path.replace(consts.ORIGINAL_IMAGES_DIRECTORY, consts.COLORED_IMAGES_DIRECTORY)

    picture = plt.imread(colored_path)
    # plt.imshow(picture)
    # plt.show()
    if not (picture[candidate_y][candidate_x] == orange_color).all():
        return 0
    # finds the boundaries of the orange rectangle
    up_boundary_y, down_boundary_y, left_boundary_x, right_boundary_x = \
        find_orange_rectangle_boundaries(picture, candidate_x, candidate_y, up_boundary_y, down_boundary_y,
                                         left_boundary_x, right_boundary_x, orange_color)
    orange_rectangle_area = (down_boundary_y - up_boundary_y) * (right_boundary_x - left_boundary_x)
    orange_rectangle = Rectangle(left_boundary_x, up_boundary_y, right_boundary_x, down_boundary_y)
    crop_rectangle = Rectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    intersection = orange_rectangle & crop_rectangle

    if intersection is None:
        return 0
    intersection_rectangle_area = (intersection.x2 - intersection.x1) * (intersection.y2 - intersection.y1)
    given_rectangle_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    intersection_with_orange_ratio = intersection_rectangle_area / orange_rectangle_area
    intersection_with_given_ratio = intersection_rectangle_area / given_rectangle_area

    if (intersection_with_given_ratio >= 0.12 and intersection_with_orange_ratio >= 0.4) or \
            (intersection_with_given_ratio >= 0.85 and intersection_with_orange_ratio >= 0.15):
        return 1  # True
    elif intersection_with_orange_ratio < 0.2 or intersection_with_given_ratio < 0.1:
        return 0  # False
    return 2
    # return (0 if (intersection_with_orange_ratio < 0.2 or intersection_with_given_ratio < 0.2) else 2
    #         if (0.2 <= intersection_with_orange_ratio < 0.65 or 0.2 <= intersection_with_given_ratio < 0.65)
    #         else 1)


def crop_images(top_left_x: int, top_left_y: int, bottom_right_x: int, bottom_right_y: int, image_path: str,
                is_green: bool, is_traffic_light: bool, number: int) -> str:
    """Cropping a rectangle which suspected as traffic light from am image.
    Consider (top_left_x,top_left_y) as the top-left vertex
    and (bottom_right_x,bottom_right_y) as the bottom-right vertex of a rectangle region within an image.
    :param top_left_x: X coordinate of top left vertex.
    :param top_left_y: Y coordinate of top left vertex.
    :param bottom_right_x: X coordinate of bottom right vertex.
    :param bottom_right_y: Y coordinate of bottom right vertex.
    :param image_path: The path to the wanted image
    :param is_green: True = green, False = Red
    :param is_traffic_light: True = Is traffic light, False = Not traffic light
    :param number: For cropped image number.
    :return: Cropped image path.
    """

    img = cv2.imread(image_path)
    number_str = "0" * (5 - len(str(number))) + str(number)
    crop_img = img[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
    cropped_image_name = "..\\cropped_images\\" + re.search(r"([^\\]+)leftImg8bit.png$", image_path).group(1)
    cropped_image_name += (('gT' if is_traffic_light else 'gF') if is_green else ('rT' if is_traffic_light else 'rF')) \
                          + "_" + number_str + ".png"
    crop_img = cv2.resize(crop_img, dsize=consts.CROPPED_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    # resized_image = crop_img.resize
    cv2.imwrite(fr"{cropped_image_name}", crop_img)
    return cropped_image_name


def get_traffic_light_rectangle(center_x: float, center_y: float, zoom: float, is_green: bool,
                                height: int, width: int) -> \
        (int, int, int, int):
    """ Receive the point of red light that has been found and the zoom. Return two points of a rectangle where the
    traffic light should be in the picture.
    :param center_x: The x value of the red light.
    :param center_y: The y value of the red light.
    :param zoom: The zoom that done to the image to find this point.
    :return: Top left point and button right point of a rectangle.
    """
    if is_green:
        top_left_offset = {0.5: (-14, -46), 0.25: (-22, -96), 0.125: (-22, -110), 0.0625: (-22, -110)}
        bottom_right_offset = {0.5: (13, 10), 0.25: (24, 20), 0.125: (28, 18), 0.0625: (28, 18)}
    else:
        top_left_offset = {0.5: (-10, -10), 0.25: (-17, -24), 0.125: (-15, -19), 0.0625: (-25, -37)}
        bottom_right_offset = {0.5: (10, 24), 0.25: (33, 93), 0.125: (27, 83), 0.0625: (58, 141)}
    value = top_left_offset.get(zoom)
    steps_to_top_left = value if value else (0, 0)
    value = bottom_right_offset.get(zoom)
    steps_to_bottom_right = value if value else (0, 0)
    top_left_x = max(center_x + steps_to_top_left[0], 0)
    top_left_y = max(center_y + steps_to_top_left[1], 0)
    # We should put here minimum between this result and the size of the picture.
    bottom_right_x = min(center_x + steps_to_bottom_right[0], width)
    bottom_right_y = min(center_y + steps_to_bottom_right[1], height)

    return int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y)


def write_to_new_table(number: int, x: int, y: int, top_left_x: int, top_left_y: int, bottom_right_x: int,
                       bottom_right_y: int, original_path: str, color: str, new_table: pd.DataFrame,
                       is_green: bool) -> None:
    """ Write new row to the new table. The row include details about the cropped image.
    :param number: Row's index.
    :param x: X center.
    :param y: Y center.
    :param top_left_x: X coordinate of top left vertex.
    :param top_left_y: Y coordinate of top left vertex.
    :param bottom_right_x: X coordinate of bottom right vertex.
    :param bottom_right_y: Y coordinate of bottom right vertex.
    :param original_path: Image's path.
    :param color: The color of the traffic light.
    :param new_table: The table that we should add it a new row.
    :param is_green:
    :return: None
    """

    rectangle_type_result = check_rectangle_in_color_image(x, y, top_left_x, top_left_y, bottom_right_x,
                                                           bottom_right_y, original_path)
    is_true = True if rectangle_type_result == 1 else False
    is_ignore = True if rectangle_type_result == 2 else False
    crop_path = crop_images(top_left_x, top_left_y, bottom_right_x, bottom_right_y, original_path, is_green, is_true,
                            number)
    new_table.loc[len(new_table)] = [number, is_true, is_ignore, crop_path, original_path, top_left_x, bottom_right_x,
                                     top_left_y, bottom_right_y, color]


def main_crop_image(image_path: str, image_df: pd.DataFrame, new_table: pd.DataFrame) -> None:
    row_num = 0
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image_df = image_df.dropna()
    for _, row in image_df.iterrows():
        if row['x'] == 'NaN' or row['y'] == 'NaN':
            continue
        is_green = row['col'] == 'g'
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = \
            get_traffic_light_rectangle(row['x'], row['y'], row['zoom'], is_green, height, width)
        write_to_new_table(row_num, row['x'], row['y'], top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                           image_path, row['col'], new_table, is_green)
        # top_left_x, top_left_y, bottom_right_x, bottom_right_y = \
        #     get_red_traffic_light_rectangle(row['x'], row['y'], row['zoom']) if row['col'] == 'r' else (0, 0, 0, 0) # get_green_traffic_light_rectangle(row['x'], row['y'], row['zoom'])
        # write_to_new_table(row_num, row['x'], row['y'], top_left_x, top_left_y, bottom_right_x, bottom_right_y,
        #                    image_path, row['col'], new_table, row['col'] == 'g')
        row_num += 1


if __name__ == '__main__':
    image_path = "ulm_000024_000019_leftImg8bit.png"
    df = pd.read_hdf("attention_results.h5")
    print(type(df))
    pd.set_option('display.max_rows', None)
    # print(df.loc[df["col"] == "r"])
    picture_df = df.loc[(df["col"] == "g") & (df["path"] == image_path)]
    print(df.loc[(df["col"] == "g") & (df['zoom'] == 0.0625)])
    print(picture_df)
    image = Image.open(image_path)
    print(type(image))
    # height, width, _ = image
    # plt.imshow(image)
    # plt.show()
    for index, row in picture_df.iterrows():
        # image = np.array(Image.open("aachen_000010_000019_leftImg8bit.png"))
        # print(f"point = {row['x'] * float(row['zoom'])} , {row['y'] * float(row['zoom'])}")
        # new_size = (int(float(image.size[0]) * float(row["zoom"])), int(float(image.size[1]) * float(row["zoom"])))
        # print(image.shape[1])
        # img = image.resize(new_size)
        # plt.imshow(img)
        # plt.show()

        plt.figure(56)
        plt.clf()
        h = plt.subplot(111)
        plt.imshow(image)
        plt.figure(57)
        plt.clf()
        plt.subplot(111, sharex=h, sharey=h)
        plt.imshow(image)
        red_x, red_y, green_x, green_y = get_traffic_light_rectangle(row['x'], row['y'], row['zoom'], True, image.height, image.width)
        plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
        plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
        plt.show()