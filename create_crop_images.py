import pandas as pd
import cv2
import matplotlib as plt


def get_green_traffic_light_rectangle(center_x: float, center_y: float, zoom: float) -> \
        tuple[float, float, float, float]:
    """ Receive the point of red light that has been found and the zoom. Return two points of a rectangle where the
    traffic light should be in the picture..
    :param center_x: The x value of the red light.
    :param center_y: The y value of the red light.
    :param zoom: The zoom that done to the image to find this point.
    :return: Top left point and button right point of a rectangle.
    """
    top_left_offset = {0.5: (-5.5, -5), 0.25: (-12.5, -19.5), 0.125: (-10.5, -14), 0.0625: (-20, -32)}
    button_right_offset = {0.5: (5, 19), 0.25: (28.5, 88), 0.125: (22, 78), 0.0625: (53, 136)}
    value = top_left_offset.get(zoom)
    steps_to_top_left = value if value else (0, 0)
    value = button_right_offset.get(zoom)
    steps_to_button_right = value if value else (0, 0)
    top_left_x = max(center_x + steps_to_top_left[0], 0)
    top_left_y = max(center_y + steps_to_top_left[1], 0)
    # We should put here minimum between this result and the size of the picture.
    button_right_x = center_x + steps_to_button_right[0]
    button_right_y = center_y + steps_to_button_right[1]

    return top_left_x, top_left_y, button_right_x, button_right_y


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
        red_x, red_y, green_x, green_y = get_green_traffic_light_rectangle(row['x'], row['y'], row['zoom'])
        plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
        plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
        plt.show()