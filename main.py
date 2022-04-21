import cv2
import numpy as np

from get_tray import ProcessImage, CutOutImage, RotateImage
from perspective_transform import PerspectiveTransformer


pers_num_path = r"config\pers_num.npy"

for i in range(4, 8):
    img_bgr = cv2.imread(r"config/image/{}.bmp".format(i+1))
    pts = np.load(pers_num_path)
    height, width = img_bgr.shape[:2]
    transformer = PerspectiveTransformer(width, height, pts)
    process = ProcessImage(kernel=21)
    cut_out = CutOutImage()
    rotate = RotateImage(rect_size=100)

    img_pers = transformer.transform(img_bgr)
    img_invert = process.get_binary_image_from_img_bgr(img_pers)
    img_rot_cut = cut_out.get_rot_cut_image_from_binary_image(img_invert, img_pers)
    img_after_rotated = rotate.rotate_image(img_rot_cut)

    # cv2.imwrite(r"config/result_image/{}.bmp".format(i+1), img_after_rotated)
    cv2.namedWindow("img_dr{}".format(i+1), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img_dr{}".format(i+1), 1000, 1000)
    cv2.imshow("img_dr{}".format(i+1), img_after_rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
