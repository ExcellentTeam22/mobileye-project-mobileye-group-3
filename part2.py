import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd


class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1<x2 and y1<y2:
            return type(self)(x1, y1, x2, y2)
    __and__ = intersection

    def difference(self, other):
        inter = self&other
        if not inter:
            yield self
            return
        xs = {self.x1, self.x2}
        ys = {self.y1, self.y2}
        if self.x1 < other.x1 < self.x2: xs.add(other.x1)
        if self.x1<other.x2<self.x2: xs.add(other.x2)
        if self.y1<other.y1<self.y2: ys.add(other.y1)
        if self.y1<other.y2<self.y2: ys.add(other.y2)
        for (x1, x2), (y1, y2) in itertools.product(
            pairwise(sorted(xs)), pairwise(sorted(ys))
        ):
            rect = type(self)(x1, y1, x2, y2)
            if rect!=inter:
                yield rect
    __sub__ = difference

    def __init__(self, x1, y1, x2, y2):
        if x1>x2 or y1>y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self)==tuple(other)
    def __ne__(self, other):
        return not (self==other)

    def __repr__(self):
        return type(self).__name__+repr(tuple(self))


def pairwise(iterable):
    # https://docs.python.org/dev/library/itertools.html#recipes
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


orange_numbers = np.array([0.98039216, 0.6666667, 0.11764706, 1.], dtype='float32')


def check_rectangle_in_color_image(candidate_x, candidate_y, x0, y0, x1, y1):
    up_boundary_y = candidate_y
    down_boundary_y = candidate_y

    left_boundary_x = candidate_x
    right_boundary_x = candidate_x
    picture = plt.imread(
        r"C:\Users\97252\bootcamp\projects\mobileye\gtFine_trainvaltest\gtFine\train\aachen\aachen_000008_000019_gtFine_color.png")
    plt.imshow(picture)

    if not (picture[candidate_y][candidate_x] == orange_numbers).all():
        print("not traffic light")  # false
    else:  # finds the bounders of the orange rectangle
        while (picture[up_boundary_y][candidate_x] == orange_numbers).all():
            up_boundary_y -= 1
        while (picture[down_boundary_y][candidate_x] == orange_numbers).all():
            down_boundary_y += 1
        while (picture[candidate_y][left_boundary_x] == orange_numbers).all():
            left_boundary_x -= 1
        while (picture[candidate_y][right_boundary_x] == orange_numbers).all():
            right_boundary_x += 1
        orange_rectangle_area = (down_boundary_y - up_boundary_y) * (right_boundary_x - left_boundary_x)
        orange_rectangle = Rectangle(left_boundary_x, up_boundary_y, right_boundary_x, down_boundary_y)
        crop_rectangle = Rectangle(x0, y0, x1, y1)
        intersection = orange_rectangle & crop_rectangle
        if intersection is None:
            print("The rectangle outside the traffic light totally")  # ignore
        else:
            intersection_rectangle_area = (intersection.x2 - intersection.x1) * (intersection.y2 - intersection.y1)
            given_rectangle_area = (x1 - x0) * (y1 - y0)
            if intersection_rectangle_area / orange_rectangle_area < 0.5:
                print("The intersection with the orange rectangle too little")  # ignore
            elif intersection_rectangle_area / given_rectangle_area < 0.5:
                print("The intersection with the given rectangle too little")  # ignore
            else:
                print("I'm traffic light")  # true


check_rectangle_in_color_image(550, 571, 985.0, 234.0, 1003.0, 272.0)  # not traffic light at all
check_rectangle_in_color_image(1006, 252, 985.0, 234.0, 1003.0, 272.0)  # the intersection with the orange rectangle too little
check_rectangle_in_color_image(1006, 252, 966.0, 246.0, 1038.0, 291.0)  # the intersection with the given rectangle too little
check_rectangle_in_color_image(1006, 252, 955.0, 309.0, 1018.0, 348.0)  # the rectangle outside the traffic light totally
check_rectangle_in_color_image(1006, 252, 997.0, 239.0, 1012.0, 268.0)  # I'm traffic light


def crop_images():
    return "cbla"


def write_to_new_table(number: int, x, y, x0, y0, x1, y1, original_path: str, color: str):
    new_table = pd.DataFrame(columns=['seq',  'is_true', 'is_ignore', 'crop_path', 'original_path', 'x0', 'x1', 'y0', 'y1', 'col'])
    print(new_table)
    is_true = False
    is_ignore = False
    if check_rectangle_in_color_image(x, y, x0, y0, x1, y1) == 0:  # false
        is_true = False
        is_ignore = False
    elif check_rectangle_in_color_image(x, y, x0, y0, x1, y1) == 1:  # true
        is_true = True
        is_ignore = False
    elif check_rectangle_in_color_image(x, y, x0, y0, x1, y1) == 2:  # ignore
        is_true = False
        is_ignore = True
    crop_path = crop_images()
    new_table = new_table.append({'seq': number, 'is_true': is_true, 'is_ignore': is_ignore, 'crop_path': crop_path,
                                  'original_path': original_path, 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'col': color},
                                 ignore_index=True)
    new_table.to_hdf('new_table.h5', key='new_table', mode='w')


write_to_new_table(number=0, x=1006, y=252, x0=997.0, y0=239.0, x1=1012.0, y1=268.0, original_path="bla", color='r')

def pass_all_old_table():
    table = pd.read_hdf("attention_results.h5")
    # pd.set_option("display.max_rows", None)  # display all the rows of the table
    print(table)
    for i in range(len(table)):
        print(i, end=' ')



pass_all_old_table()
