import cv2
import numpy as np

from get_tray import GetTray
from perspective_transform import PerspectiveTransformer


pers_num_path = r"config\pers_num.npy"

for i in range(4, 8):
    img_bgr = cv2.imread(r"config/image/{}.bmp".format(i+1))
    pts = np.load(pers_num_path)
    height, width = img_bgr.shape[:2]
    transformer = PerspectiveTransformer(width, height, pts)
    get_tray = GetTray()

    img_pers = transformer.transform(img_bgr)
    img_invert = get_tray.get_binary_image_from_img_bgr(img_pers, 21)
    img_rot_cut = get_tray.get_rot_cut_image_from_binary_image(img_invert, img_pers)
    index = get_tray.get_max_value_index_in_4_corner(img_rot_cut)
    img_after_rotated = get_tray.rotate_image(index, img_rot_cut)

    # cv2.imwrite(r"config/result_image/_{}.bmp".format(i+1), img_rot_cut)
    cv2.namedWindow("img_dr{}".format(i+1), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img_dr{}".format(i+1), 1000, 1000)
    cv2.imshow("img_dr{}".format(i+1), img_after_rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
