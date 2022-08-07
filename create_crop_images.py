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


def check_rectangle_in_color_image(candidate_x: int, candidate_y: int, top_left_x: float, top_left_y: float,
                                   bottom_right_x: float, bottom_right_y: float, image_path: str) -> int:
    """
    """
    orange_color = np.array([0.98039216, 0.6666667, 0.11764706, 1.], dtype='float32')
    candidate_y = int(candidate_y)
    candidate_x = int(candidate_x)
    up_boundary_y = candidate_y
    down_boundary_y = candidate_y

    left_boundary_x = candidate_x
    right_boundary_x = candidate_x
    colored_path = image_path.replace('_leftImg8bit', '_gtFine_color')
    colored_path = colored_path.replace('leftImg8bit_trainvaltest', 'gtFine')
    # picture = plt.imread(r"aachen_000008_000019_gtFine_color.png")
    picture = plt.imread(colored_path)
    # plt.imshow(picture)
    # plt.show()
    if not (picture[int(candidate_y)][int(candidate_x)] == orange_color).all():
        # print("not traffic light")  # false
        return 0
    else:  # finds the bounders of the orange rectangle
        while up_boundary_y >= 0 and (picture[up_boundary_y][candidate_x] == orange_color).all():
            up_boundary_y -= 1
        while (picture[down_boundary_y][candidate_x] == orange_color).all():
            down_boundary_y += 1
        while left_boundary_x >= 0 and (picture[candidate_y][left_boundary_x] == orange_color).all():
            left_boundary_x -= 1
        while (picture[candidate_y][right_boundary_x] == orange_color).all():
            right_boundary_x += 1
        orange_rectangle_area = (down_boundary_y - up_boundary_y) * (right_boundary_x - left_boundary_x)
        orange_rectangle = Rectangle(left_boundary_x, up_boundary_y, right_boundary_x, down_boundary_y)
        crop_rectangle = Rectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        intersection = orange_rectangle & crop_rectangle
        if intersection is None:
            # print("The rectangle outside the traffic light totally")  # ignore
            return 0
        else:
            intersection_rectangle_area = (intersection.x2 - intersection.x1) * (intersection.y2 - intersection.y1)
            given_rectangle_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
            if intersection_rectangle_area / orange_rectangle_area < 0.5:
                # print("The intersection with the orange rectangle too little")  # ignore
                return 2
            elif intersection_rectangle_area / given_rectangle_area < 0.5:
                # print("The intersection with the given rectangle too little")  # ignore
                return 2
            else:
                # print("I'm traffic light")  # true
                return 1


def crop_images(top_left_x:int, top_left_y:int, bottom_right_x:int, bottom_right_y:int, image_path: str, is_green: bool,
                is_traffic_light: bool):
    """
        Cropping a rectangle which suspected as traffic light from am image.
        Consider (top_left_x,top_left_y) as the top-left vertex
        and (bottom_right_x,bottom_right_y) as the bottom-right vertex of a rectangle region within an image.
        Arguments:
            (top_left_x, top_left_y): Top-left vertex
            (bottom_right_x, bottom_right_y): Bottom-right vertex
            image_path: The path to the wanted image
            is_green: True = green, False = Red
            is_traffic_light: True = Is traffic light, False = Not traffic light
        Returns:
            none
        """
    img = cv2.imread(image_path)
    crop_img = img[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
    cropped_image_name = r"..\\cropped_images\\" + image_path.split("left")[0]
    cropped_image_name += (r'gT_00000.png' if is_traffic_light else r'gF_00001.png') if is_green else \
        (r'rT_00000.png' if is_traffic_light else r'rF_00001.png')
    if not cv2.imwrite(fr"{cropped_image_name}", crop_img):
        raise Exception("wlefn")
    return cropped_image_name


def get_red_traffic_light_rectangle(center_x: float, center_y: float, zoom: float) -> \
        (int, int, int, int):
    """ Receive the point of red light that has been found and the zoom. Return two points of a rectangle where the
    traffic light should be in the picture.
    :param center_x: The x value of the red light.
    :param center_y: The y value of the red light.
    :param zoom: The zoom that done to the image to find this point.
    :return: Top left point and button right point of a rectangle.
    """
    top_left_offset = {0.5: (-5, -5), 0.25: (-12, -19), 0.125: (-10, -14), 0.0625: (-20, -32)}
    bottom_right_offset = {0.5: (5, 19), 0.25: (28, 88), 0.125: (22, 78), 0.0625: (53, 136)}
    value = top_left_offset.get(zoom)
    steps_to_top_left = value if value else (0, 0)
    value = bottom_right_offset.get(zoom)
    steps_to_bottom_right = value if value else (0, 0)
    top_left_x = max(center_x + steps_to_top_left[0], 0)
    top_left_y = max(center_y + steps_to_top_left[1], 0)
    # We should put here minimum between this result and the size of the picture.
    bottom_right_x = center_x + steps_to_bottom_right[0]
    bottom_right_y = center_y + steps_to_bottom_right[1]

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def write_to_new_table(number: int, x, y, top_left_x, top_left_y, bottom_right_x, bottom_right_y, original_path: str,
                       color: str, new_table: pd.DataFrame, is_green: bool) -> None:

    rectangle_type_result = check_rectangle_in_color_image(x, y, top_left_x, top_left_y, bottom_right_x,
                                                           bottom_right_y, original_path)
    is_true = True if rectangle_type_result == 1 else False
    is_ignore = True if rectangle_type_result == 2 else False
    crop_path = crop_images(top_left_x, top_left_y, bottom_right_x, bottom_right_y, original_path, is_green, is_true)
    new_table.loc[len(new_table)] = [number, is_true, is_ignore, crop_path, original_path, top_left_x, bottom_right_x,
                                     top_left_y, bottom_right_y, color]


def main_crop_image(image_path: str, image_df: pd.DataFrame, new_table:pd.DataFrame) -> None:
    row_num = 0
    for index, row in image_df.iterrows():
        if row['col'] == 'r':
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = \
                get_red_traffic_light_rectangle(row['x'], row['y'], row['zoom'])
            write_to_new_table(row_num, row['x'], row['y'], top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                               image_path, row['col'], new_table, row['col'] == 'g')
        # top_left_x, top_left_y, bottom_right_x, bottom_right_y = \
        #     get_red_traffic_light_rectangle(row['x'], row['y'], row['zoom']) if row['col'] == 'r' else (0, 0, 0, 0) # get_green_traffic_light_rectangle(row['x'], row['y'], row['zoom'])
        # write_to_new_table(row_num, row['x'], row['y'], top_left_x, top_left_y, bottom_right_x, bottom_right_y,
        #                    image_path, row['col'], new_table, row['col'] == 'g')
        row_num += 1


if __name__ == '__main__':
    image_path = "dusseldorf_000068_000019_leftImg8bit.png"
    df = pd.read_hdf("attention_results.h5")
    print(type(df))
    pd.set_option('display.max_rows', None)
    # print(df.loc[df["col"] == "r"])
    picture_df = df.loc[(df["col"] == "r") & (df["path"] == image_path)]
    print(df.loc[(df["col"] == "r") & (df["zoom"] == 0.0625)])
    print(picture_df)
    image = Image.open(image_path)
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
        red_x, red_y, green_x, green_y = get_red_traffic_light_rectangle(row['x'], row['y'], row['zoom'])
        plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
        plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
        plt.show()