"""
img[ul_h:lr_h, ul_w:lr_w]
"""
import cv2


class Matrix:
    def __init__(self):
        """分割のための座標を取得するクラス"""
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


class Separate:
    def __init__(self):
        """画像をポケットサイズで分けるクラス"""
        pass

    @staticmethod
    def get_pocket_size_image(img, coordinate, w_interval, h_interval):
        """座標と間隔で画像を分割"""
        x, y = round(coordinate[0]), round(coordinate[1])
        return img[y:y+h_interval, x:x+w_interval]

    def func(self):
        """リストの個数分、画像取得、二値化して白があるかどうか判定する"""
        pass


if __name__ == '__main__':
    _ul = (95, 115)
    _lr = (1495, 1525)
    _pocket_num = 20
    _img = cv2.imread(r"config/result_image/20.bmp")

    mat = Matrix()
    sep = Separate()
    _w_interval, _h_interval = mat.get_interval(_ul, _lr, _pocket_num)
    _coordinate_list = mat.get_upper_left_coordinates(_ul, _lr, _pocket_num)
    _coordinate = _coordinate_list[0]
    _img_sep = sep.get_pocket_size_image(_img, _coordinate, _w_interval, _h_interval)

    cv2.namedWindow("img_rect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img_rect", 100, 100)
    cv2.imshow("img_rect", _img_sep)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
