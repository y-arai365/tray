import cv2


class Matrix:
    def __init__(self):
        """画像をポケットごとに分ける"""
        pass

    def draw_vertical_lines(self, img, start, stop, pocket_num, interval):
        """縦線を描画する"""
        (a, b) = start
        (c, d) = stop
        drawn_img = self._draw_line(img, start, stop)
        for i in range(pocket_num):
            start = (round(a+((i+1)*interval)), b)
            stop = (round(c+((i+1)*interval)), d)
            drawn_img = self._draw_line(drawn_img, start, stop)
        return drawn_img

    def draw_horizontal_lines(self, img, start, stop, pocket_num, interval):
        """横線を描画する"""
        (a, b) = start
        (c, d) = stop
        drawn_img = self._draw_line(img, start, stop)
        for i in range(pocket_num):
            start = (a, round(b+((i+1)*interval)))
            stop = (c, round(d+((i+1)*interval)))
            drawn_img = self._draw_line(drawn_img, start, stop)
        return drawn_img

    @staticmethod
    def _draw_line(img, start, stop):
        return cv2.line(img, start, stop, (0, 0, 255), 2)

    @staticmethod
    def get_interval(ul, lr, pocket_num):
        """2点とポケット数からintervalを求める"""
        (ul_w, ul_h) = ul
        (lr_w, lr_h) = lr
        w_length = lr_w - ul_w
        h_length = lr_h - ul_h
        w_interval = w_length/pocket_num
        h_interval = h_length/pocket_num
        return w_interval, h_interval


if __name__ == '__main__':
    """8.16.20.23.27.30.35, (95, 115), (1495, 1525)"""
    """5.13.19.24.(28).31.36, (110, 105), (1520, 1510)"""
    """6.14.18.21.25.32.33, (90, 80), (1515, 1510)?"""
    """7.15.17.22.26.29.34, (90, 90), (1505, 1500)?"""
    mat = Matrix()
    _img = cv2.imread(r"config/result_image/30.bmp")
    _h, _w = _img.shape[:2]

    _ul = (130, 140)
    _lr = (1480, 1490)
    _pocket_num = 20

    _img = cv2.circle(_img, _ul, 5, (255, 255, 0), 2)
    _img = cv2.circle(_img, _lr, 5, (255, 255, 0), 2)

    _start_w = (_ul[0], 0)
    _stop_w = (_ul[0], _h)
    _start_h = (0, _ul[1])
    _stop_h = (_w, _ul[1])

    _w_interval, _h_interval = mat.get_interval(_ul, _lr, _pocket_num)
    _img = mat.draw_vertical_lines(_img, _start_w, _stop_w, _pocket_num, _w_interval)
    _img = mat.draw_horizontal_lines(_img, _start_h, _stop_h, _pocket_num, _h_interval)

    cv2.namedWindow("img_rect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img_rect", 900, 900)
    cv2.imshow("img_rect", _img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
