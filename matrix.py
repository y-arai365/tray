"""
img[ul_h:lr_h, ul_w:lr_w]
"""
import cv2
import numpy as np


class Matrix:
    def __init__(self):
        """分割のための座標を取得、判定するクラス"""
        pass

    @staticmethod
    def get_interval(ul, lr, pocket_num):
        """2点とポケット数からintervalを求める"""
        (ul_w, ul_h) = ul
        (lr_w, lr_h) = lr
        w_length = lr_w - ul_w
        h_length = lr_h - ul_h
        w_interval = w_length/pocket_num
        h_interval = h_length/pocket_num
        return round(w_interval), round(h_interval)

    def get_upper_left_coordinates(self, ul, lr, pocket_num):
        """座標リストを返す"""
        (ul_w, ul_h) = ul
        w_interval, h_interval = self.get_interval(ul, lr, pocket_num)
        coordinate_list = []

        for i in range(pocket_num):  # width
            for j in range(pocket_num):  # height
                width = ul_w + (i*w_interval)
                height = ul_h + (j*h_interval)
                coordinate_list.append((width, height))
        return coordinate_list


class Judgement:
    def __init__(self):
        """画像をポケットサイズで分けるクラス"""
        pass

    def mark_blank_pocket(self, img, coordinate_list, w_interval, h_interval):
        """空のポケットを矩形表示して指摘"""
        img_copy = img.copy()
        for coordinate in coordinate_list:
            pocket_image = self._get_pocket_image(img_copy, coordinate, w_interval, h_interval)
            binary_pocket_image = self._process_img_bgr_to_img_th(pocket_image)
            result = self._judge(binary_pocket_image)
            if result == 0:
                img_copy = self._mark(img_copy, coordinate, w_interval, h_interval)
        return img_copy

    @staticmethod
    def _get_pocket_image(img, coordinate, w_interval, h_interval):
        """座標と間隔で画像を分割"""
        x, y = round(coordinate[0]), round(coordinate[1])
        return img[y:y+h_interval, x:x+w_interval]

    @staticmethod
    def _process_img_bgr_to_img_th(img):
        """BGR画像を二値化画像に変更する"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 5)
        _, binary_pocket_image = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY)
        return binary_pocket_image

    @staticmethod
    def _judge(binary_pocket_image):
        """二値化画像に白が含まれていたら255、黒のみなら0を返す"""
        return np.amax(binary_pocket_image)

    @staticmethod
    def _mark(img, coordinate, w_interval, h_interval):
        """指定範囲に矩形を描画する"""
        x, y = round(coordinate[0]), round(coordinate[1])
        return cv2.rectangle(img, (x+2, y+2), (x+w_interval-2, y+h_interval-2), (0, 0, 255), 2)


if __name__ == '__main__':
    _ul = (130, 140)
    _lr = (1480, 1490)
    _pocket_num = 20
    _img = cv2.imread(r"config/result_image/30.bmp")

    mat = Matrix()
    jud = Judgement()
    _w_interval, _h_interval = mat.get_interval(_ul, _lr, _pocket_num)
    _coordinate_list = mat.get_upper_left_coordinates(_ul, _lr, _pocket_num)

    _result_image = jud.mark_blank_pocket(_img, _coordinate_list, _w_interval, _h_interval)
    cv2.namedWindow("img_rect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img_rect", 1000, 1000)
    cv2.imshow("img_rect", _result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
