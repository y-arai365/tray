import cv2
import numpy as np

from get_tray import ProcessImage, CutOutImage, RotateImage
from matrix import Matrix, Judgement
from perspective_transform import PerspectiveTransformer


pers_num_path = r"config\pers_num.npy"

ul = (130, 140)
lr = (1480, 1490)
pocket_num = 20
threshold = 100

pts = np.load(pers_num_path)
process = ProcessImage(kernel=21)
cut_out = CutOutImage()
rotate = RotateImage(rect_size=100)
matrix = Matrix()
judge = Judgement()

for i in range(29, 30):
    img_bgr = cv2.imread(r"config/image/{}.bmp".format(i+1))
    height, width = img_bgr.shape[:2]
    transformer = PerspectiveTransformer(width, height, pts)

    img_pers = transformer.transform(img_bgr)
    img_invert = process.get_binary_image_from_img_bgr(img_pers)
    img_rot_cut = cut_out.get_rot_cut_image_from_binary_image(img_invert, img_pers)
    img_after_rotated = rotate.rotate_image(img_rot_cut)

    w_interval, h_interval = matrix.get_interval(ul, lr, pocket_num)
    coordinate_list = matrix.get_upper_left_coordinates(ul, lr, pocket_num)
    result_image = judge.mark_blank_pocket(img_after_rotated, coordinate_list, w_interval, h_interval, threshold)

    # cv2.imwrite(r"config/result_image/{}_.bmp".format(i+1), img_pers)
    # cv2.namedWindow("img_dr{}".format(i+1), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("img_dr{}".format(i+1), 900, 900)
    # cv2.imshow("img_dr{}".format(i+1), img_pers)

cv2.waitKey(0)
cv2.destroyAllWindows()
