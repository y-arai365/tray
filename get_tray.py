import cv2
import numpy as np


class GetTray:
    def __init__(self):
        pass

    def get_binary_image_from_img_bgr(self, img_bgr, kernel):
        """bgr画像から二値化反転画像を返す"""
        img_gray = self._grayscale_image(img_bgr)
        img_blur = self._blur_image(img_gray, kernel)
        return self._binarize_and_invert_image(img_blur)

    def get_rot_cut_image_from_binary_image(self, img_bi, img_pers):
        """二値化画像から輪郭を取得して、切り取る"""
        cnt, _ = self._get_min_rect(img_bi)
        center, size, deg = self._get_rect(cnt)
        return self._rot_cut(img_pers, deg, center, size)

    def get_max_value_index_in_4_corner(self, img, rect_size=100):
        """画像の4隅を取得し、その各画像の色平均のV(明度)が高いインデックス(位置を返す)"""
        h, w = img.shape[:2]
        upper_left, upper_right, lower_left, lower_right = self._get_corner_image(img, h, w, rect_size)
        upper_left_v = self._get_corner_image_value(upper_left)
        upper_right_v = self._get_corner_image_value(upper_right)
        lower_left_v = self._get_corner_image_value(lower_left)
        lower_right_v = self._get_corner_image_value(lower_right)
        corner_v_list = [upper_left_v, upper_right_v, lower_left_v, lower_right_v]
        return self._get_max_value_index(corner_v_list)

    @staticmethod
    def _get_corner_image(img, h, w, rect_size):
        """4隅の画像を取得"""
        upper_left = cv2.cvtColor(img[0:rect_size, 0:rect_size], cv2.COLOR_BGR2HSV)
        upper_right = cv2.cvtColor(img[0:rect_size, w - rect_size:w], cv2.COLOR_BGR2HSV)
        lower_left = cv2.cvtColor(img[h - rect_size:h, 0:rect_size], cv2.COLOR_BGR2HSV)
        lower_right = cv2.cvtColor(img[h - rect_size:h, w - rect_size:w], cv2.COLOR_BGR2HSV)
        return upper_left, upper_right, lower_left, lower_right

    @staticmethod
    def _get_corner_image_value(corner_image):
        """角画像のV(明度)を取得"""
        return corner_image.T[2].flatten().mean()

    @staticmethod
    def _get_max_value_index(value_list):
        """リストで一番高いV(明度)のインデックスを返す"""
        return value_list.index(max(value_list))

    @staticmethod
    def rotate_image(max_v_index, img):
        """
        インデックスの値に応じて画像を回転させて位置を合わせる

        Args:
            max_v_index (int): 0 or 1 or 2 or 3
            img (np.ndarray): 矩形切り取りした画像

        Returns:
            img (np.ndarray): 向きを合わせた画像
        """
        if max_v_index == 0:
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif max_v_index == 2:
            img_rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif max_v_index == 3:
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img_rotated = img
        return img_rotated

    @staticmethod
    def _grayscale_image(img_bgr):
        """bgr画像をグレイスケールに変換"""
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _blur_image(img_gray, kernel):
        return cv2.medianBlur(img_gray, kernel)

    @staticmethod
    def _binarize_and_invert_image(img_blur):
        """画像を二値化して返す"""
        _, img_bi = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return img_bi

    # @staticmethod
    # def invert_image(img_bi):
    #     """画像の白黒反転"""
    #     return cv2.bitwise_not(img_bi)

    def _get_min_rect(self, img_bi):
        """画像から最小外接矩形を取得"""
        contours, hierarchy = cv2.findContours(img_bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]
        cnt = self._get_rect_contour(contours)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # print(box)
        return cnt, box

    @staticmethod
    def _get_rect_contour(contours):
        """全輪郭から矩形部分の輪郭を取得する"""
        len_cnt_list = []
        for num, cnt in enumerate(contours):
            len_cnt_list.append(len(cnt))
        max_len_cnt_index = len_cnt_list.index(max(len_cnt_list))
        return contours[max_len_cnt_index]

    # @staticmethod
    # def draw_rect(img, box):
    #     return cv2.drawContours(img, [box], 0, (0, 0, 255), 4)

    @staticmethod
    def _get_rect(contour):
        center, size, deg = cv2.minAreaRect(contour)
        size = np.int0(size)
        return center, size, deg

    @staticmethod
    def _rot_cut(src_img, deg, center, size):
        """傾きを整えて画像の切り取りを行う"""
        rot_mat = cv2.getRotationMatrix2D(center, deg, 1.0)
        rot_mat[0][2] += -center[0] + size[0] / 2  # -(元画像内での中心位置)+(切り抜きたいサイズの中心)
        rot_mat[1][2] += -center[1] + size[1] / 2  # 同上
        return cv2.warpAffine(src_img, rot_mat, size)


if __name__ == '__main__':
    image_path = r"config/image/1.bmp"
    _img = cv2.imread(image_path)
    get_tray = GetTray()

    _img_bi = get_tray.get_binary_image_from_img_bgr(_img, 21)
    _img_bi = get_tray.get_rot_cut_image_from_binary_image(_img_bi, _img)

    cv2.namedWindow("img_rect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img_rect", 1200, 900)
    cv2.imshow("img_rect", _img_bi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
