"""球場參考資料：提供標準網球場幾何資訊。"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class CourtReference:
    """
    Court reference model

    提供網球場線條、關鍵點與遮罩等靜態資訊。
    """
    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)
        
        self.key_points = [*self.baseline_top, *self.baseline_bottom, 
                          *self.left_inner_line, *self.right_inner_line,
                          *self.top_inner_line, *self.bottom_inner_line,
                          *self.middle_line]
        # key_points 順序與參考圖一致，供單應矩陣應用
        
        self.border_points = [*self.baseline_top, *self.baseline_bottom[::-1]]

        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                               self.right_inner_line[1]],
                           3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                               self.right_court_line[1]],
                           4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                               self.right_inner_line[1]],
                           5: [*self.top_inner_line, *self.bottom_inner_line],
                           6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
                           7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
                           8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                               self.right_court_line[1]],
                           9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
                               self.left_inner_line[1]],
                           10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0],
                                self.middle_line[1]],
                           11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1],
                                self.bottom_inner_line[1]],
                           12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}
        self.line_width = 1
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2
        self.court = self.build_court_reference()

        # self.court = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)

    def build_court_reference(self):
        """
        Create court reference image using the lines positions

        以線段座標建立灰階場地示意圖，後續可做小地圖繪製。
        """
        court = np.zeros((self.court_height + 2 * self.top_bottom_border, self.court_width + 2 * self.right_left_border), dtype=np.uint8)
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        cv2.line(court, *self.net, 1, self.line_width)
        cv2.line(court, *self.top_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
        cv2.line(court, *self.left_court_line, 1, self.line_width)
        cv2.line(court, *self.right_court_line, 1, self.line_width)
        cv2.line(court, *self.left_inner_line, 1, self.line_width)
        cv2.line(court, *self.right_inner_line, 1, self.line_width)
        cv2.line(court, *self.middle_line, 1, self.line_width)
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        # court = cv2.dilate(court, np.ones((7, 7), dtype=np.uint8))
        # plt.imsave('court_configurations/court_reference.png', court, cmap='gray')
        # self.court = court
        return court

    def get_important_lines(self):
        """
        Returns all lines of the court

        回傳場地所有主線段，便於外部使用。
        """
        lines = [*self.baseline_top, *self.baseline_bottom, *self.net, *self.left_court_line, *self.right_court_line,
                 *self.left_inner_line, *self.right_inner_line, *self.middle_line,
                 *self.top_inner_line, *self.bottom_inner_line]
        return lines

    def get_extra_parts(self):
        """取得球場上下延伸線段的端點座標。"""
        parts = [self.top_extra_part, self.bottom_extra_part]
        return parts

    def save_all_court_configurations(self):
        """
        Create all configurations of 4 points on court reference

        將所有四點組合標註在場地圖上並輸出為影像檔。
        """
        for i, conf in self.court_conf.items():
            c = cv2.cvtColor(255 - self.court, cv2.COLOR_GRAY2BGR)
            for p in conf:
                c = cv2.circle(c, p, 15, (0, 0, 255), 30)
            cv2.imwrite(f'court_configurations/court_conf_{i}.png', c)

    def get_court_mask(self, mask_type=0):
        """
        Get mask of the court

        根據指定型態回傳不同區域的遮罩。
        """
        mask = np.ones_like(self.court)
        if mask_type == 1:  # Bottom half court
            # 只保留下半場，方便標記底線選手
            # mask[:self.net[0][1] - 1000, :] = 0
            mask[:self.net[0][1], :] = 0
        elif mask_type == 2:  # Top half court
            # 只保留上半場
            mask[self.net[0][1]:, :] = 0
        elif mask_type == 3: # court without margins
            # 去除外圍邊界，留下正式場地範圍
            mask[:self.baseline_top[0][1], :] = 0
            mask[self.baseline_bottom[0][1]:, :] = 0
            mask[:, :self.left_court_line[0][0]] = 0
            mask[:, self.right_court_line[0][0]:] = 0
        return mask


if __name__ == '__main__':
    c = CourtReference()
    c.build_court_reference()
