import cv2


def crop_images(x1, y1, x2, y2, image_path: str, is_green: bool, is_traffic_light: bool):
    """
        Cropping a rectangle which suspected as traffic light from am image.
        Consider (x1,y1) as the top-left vertex
        and (x2,y2) as the bottom-right vertex of a rectangle region within an image.
        Arguments:
            (x1, y1): Top-left vertex
            (x2, y2): Bottom-right vertex
            image_path: The path to the wanted image
            is_green: True = green, False = Red
            is_traffic_light: True = Is traffic light, False = Not traffic light
        Returns:
            none
        """
    img = cv2.imread(image_path)
    crop_img = img[y1:y2, x1:x2]
    cropped_image_name = image_path.split("left")

    if is_green:
        if is_traffic_light:
            cropped_image_name = cropped_image_name[0] + 'gT_0000.png'
        else:
            cropped_image_name = cropped_image_name[0] + 'gF_0001.png'

    else:
        if is_traffic_light:
            cropped_image_name = cropped_image_name[0] + 'rT_0000.png'
        else:
            cropped_image_name = cropped_image_name[0] + 'rF_0001.png'
    print(cropped_image_name)

    cv2.imwrite(fr"Cropped_Images\{cropped_image_name}", crop_img)
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)



# berlin_000035_000019_leftImg8bit.png
# 1048, 314
# 1063, 365
# ############################################3
# berlin_000017_000019_leftImg8bit.png
# 1102, 247
# 1120, 289

crop_images(1102, 247, 1120, 289, "berlin_000017_000019_leftImg8bit.png", True, False)
