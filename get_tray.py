import cv2
import numpy as np


class ProcessImage:
    def __init__(self, kernel):
        """
        画像に処理を加えるクラス

        Args:
            kernel (int): 画像のぼかし具合
        """
        self.kernel = kernel

    def get_binary_image_from_img_bgr(self, img_bgr):
        """
        bgr画像から二値化反転画像を返す

        Args:
            img_bgr (img_bgr): 射影変換済みの画像

        Returns:
            img_th: 二値化画像

        """
        img_gray = self._grayscale_image(img_bgr)
        img_blur = self._blur_image(img_gray, self.kernel)
        return self._binarize_and_invert_image(img_blur)

    @staticmethod
    def _grayscale_image(img_bgr):
        """bgr画像をグレイスケールに変換"""
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _blur_image(img_gray, kernel):
        """グレイスケール画像にぼかしを入れる"""
        return cv2.medianBlur(img_gray, kernel)

    @staticmethod
    def _binarize_and_invert_image(img_blur):
        """画像を二値化して返す"""
        _, img_bi = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return img_bi


class CutOutImage:
    def __init__(self):
        """画像を切り取るクラス"""
        pass

    def get_rot_cut_image_from_binary_image(self, img_bi, img_pers):
        """
        二値化画像から輪郭を取得して、切り取る

        Args:
            img_bi (img_th): 射影変換後に二値化した画像
            img_pers (img_bgr): img_bgr (img_bgr): 射影変換後の画像

        Returns:
            img_bgr: 矩形に沿って切り出して、向きを整えた画像
        """
        cnt, _ = self._get_min_rect(img_bi)
        center, size, deg = self._get_rect(cnt)
        return self._rot_cut(img_pers, deg, center, size)

    def _get_min_rect(self, img_bi):
        """二値化画像から最小外接矩形を取得"""
        contours, hierarchy = cv2.findContours(img_bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = self._get_rect_contour(contours)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return cnt, box

    @staticmethod
    def _get_rect_contour(contours):
        """全輪郭から矩形部分の輪郭を面積の大きさで判断して取得する"""
        len_cnt_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            len_cnt_list.append(area)
        max_len_cnt_index = len_cnt_list.index(max(len_cnt_list))
        return contours[max_len_cnt_index]

    @staticmethod
    def _get_rect(contour):
        """輪郭点からその図形の中心、大きさ、傾きを取得する"""
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


class RotateImage:
    def __init__(self, rect_size):
        """
        画像の向きを揃えるクラス

        Args:
            rect_size (int): 4隅の画像を取得するときの画像の幅と高さ
        """
        self.rect_size = rect_size

    def rotate_image(self, img):
        """
        インデックスの値に応じて画像を回転させて位置を合わせる

        Args:
            img (img_bgr): 矩形切り取りした画像

        Returns:
            img_bgr: 向きを合わせた画像
        """
        max_value_index = self._get_max_value_index_in_4_corner(img)
        if max_value_index == 0:
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif max_value_index == 2:
            img_rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif max_value_index == 3:
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img_rotated = img
        return img_rotated

    def _get_max_value_index_in_4_corner(self, img):
        """画像の4隅を取得し、その各画像の色平均のV(明度)が高いインデックス(位置を返す)"""
        h, w = img.shape[:2]
        upper_left, upper_right, lower_left, lower_right = self._get_corner_image(img, h, w)
        upper_left_v = self._get_corner_image_value(upper_left)
        upper_right_v = self._get_corner_image_value(upper_right)
        lower_left_v = self._get_corner_image_value(lower_left)
        lower_right_v = self._get_corner_image_value(lower_right)
        corner_v_list = [upper_left_v, upper_right_v, lower_left_v, lower_right_v]
        return self._get_max_value_index(corner_v_list)

    def _get_corner_image(self, img, h, w):
        """4隅の画像を取得"""
        upper_left = cv2.cvtColor(img[0:self.rect_size, 0:self.rect_size], cv2.COLOR_BGR2HSV)
        upper_right = cv2.cvtColor(img[0:self.rect_size, w - self.rect_size:w], cv2.COLOR_BGR2HSV)
        lower_left = cv2.cvtColor(img[h - self.rect_size:h, 0:self.rect_size], cv2.COLOR_BGR2HSV)
        lower_right = cv2.cvtColor(img[h - self.rect_size:h, w - self.rect_size:w], cv2.COLOR_BGR2HSV)
        return upper_left, upper_right, lower_left, lower_right

    @staticmethod
    def _get_corner_image_value(corner_image):
        """隅画像のV(明度)を取得"""
        return corner_image.T[2].flatten().mean()

    @staticmethod
    def _get_max_value_index(value_list):
        """リストで一番高いV(明度)のインデックスを返す"""
        return value_list.index(max(value_list))


if __name__ == '__main__':
    from perspective_transform import PerspectiveTransformer

    image_path = r"config/image/1.bmp"
    _img = cv2.imread(image_path)

    pers_num_path = r"config\pers_num.npy"
    pts = np.load(pers_num_path)
    height, width = _img.shape[:2]
    transformer = PerspectiveTransformer(width, height, pts)

    process = ProcessImage(21)
    cut_out = CutOutImage()
    rotate = RotateImage(100)

    _img_pers = transformer.transform(_img)
    _img_bi = process.get_binary_image_from_img_bgr(_img_pers)
    _img_rot_cut = cut_out.get_rot_cut_image_from_binary_image(_img_bi, _img_pers)
    _img_after_rotated = rotate.rotate_image(_img_rot_cut)

    cv2.namedWindow("img_rect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img_rect", 900, 900)
    cv2.imshow("img_rect", _img_after_rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
