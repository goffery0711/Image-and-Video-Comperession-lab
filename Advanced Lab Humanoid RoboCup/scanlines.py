import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import cv2
import numpy as np


def plot_groud_truth(test_img):

    test_img_label = test_img[:-3]
    test_img_label = test_img_label + 'txt'
    print(test_img_label)

    with open(test_img_label, 'r') as f:

        lines = f.readlines()

        for l in lines:
            # remove \n
            n_list = l.strip()
            # line str to list with str elements
            n_list = n_list.split(' ')
            # str to float
            n = np.array(n_list, dtype=float)

            # 3 == robot, [x, y, w, h]
            # values normalized by img width (x/512) and height (y/384)
            w = float(512)
            h = float(384)

            if n[0] == 3:
                xy = [n[1] * w, n[2] * h]
                x = w * np.array([n[1] + n[3] / 2, n[1] + n[3] / 2, n[1] - n[3] / 2, n[1] - n[3] / 2])
                y = h * np.array([n[2] + n[4] / 2, n[2] - n[4] / 2, n[2] + n[4] / 2, n[2] - n[4] / 2])
                plt.scatter(x=xy[0], y=xy[1], c='r', s=40)
                # plot frame
                plt.plot([x[0], x[1]], [y[0], y[1]], 'ro-')
                plt.plot([x[1], x[3]], [y[1], y[3]], 'ro-')
                plt.plot([x[3], x[2]], [y[3], y[2]], 'ro-')
                plt.plot([x[2], x[0]], [y[2], y[0]], 'ro-')


def plot_horizontal_scanlines(img):

    # 384, 512, 3
    [h, w, channel] = img.shape

    num_lines = h/10

    print(num_lines)


def plot_vertical_scanlines(img):

    # 384, 512, 3
    [h, w, channel] = img.shape

    num_lines = w/20

    print(num_lines)


def scan_mask(img, row_dist=7, col_dist=10, h_s=30, w_s=5):

    # 384, 512, 3
    [h, w, channel] = img.shape

    # create mask
    # each dot samples the color

    # create list of to be sampled rows
    row_list = []
    # list of to be sampled col
    col_list = []

    row = h_s
    col = w_s
    if type(h) == int and type(w) == int and col_dist > 0 and row_dist > 0:
        while row < h:
            row_list.append(row)

            #if len(row_list) % 5 == 0:
            #row_dist = row_dist+1
            # new row num
            row = row + row_dist

        # list of to be sampled col
        while col < w:
            col_list.append(col)
            col = col + col_dist
    else:
        print('cond not fulfilled')

    mask = np.zeros([h, w])
    for i in row_list:
        for j in col_list:
            mask[i, j] = 1

    return mask, row_list, col_list, h_s, w_s


def idx_change(idx_candidates, sum_list, dots, threshold):

    valid_sums = sum_list[dots]

    # initialize
    if valid_sums[0] > threshold:
        prev_cond = True
    else:
        prev_cond = False

    for idx, d in enumerate(valid_sums):

        current_cond = (d > threshold)
        if current_cond != prev_cond:
            idx_candidates.append(dots[idx])
            prev_cond = current_cond

    return idx_candidates


def get_candidatebox(filter_img, w_dots, h_s, w_s, threshold=10, window_width=100):

    [h, w] = np.shape(filter_img)
    #num_width = len(w_dots)
    window_dist = w_dots[10] - w_dots[9]
    shift_num = math.ceil((w - (window_width + w_s))/window_dist)
    print('__________')
    print(shift_num)

    # initialize
    height = [h_s, h]
    #print('height')
    #print(height)
    width = [w_s, window_width+w_s]
    #print('width')
    #print(width)
    width_list = []
    width_val = []
    for n in range(shift_num):

        width_list.append(width)
        window_img = filter_img[h_s:h, width[0]:width[1]]
        sum_window = sum(map(sum, window_img))

        width_val.append(sum_window)

        # new width
        width = [x+window_dist for x in width]

    #print(width_list)
    #print(width_val)

    # find n smallest val and its idx in list
    idx = np.argsort(width_val)
    print(idx)
    final_candidates = []
    for i in range(5):
        print(width_val[idx[i]])
        final_candidates.append(width_list[idx[i]])
    #print(width_val[idx[:5]])
    #print(width_val)


    return final_candidates, height


def get_bounding_box(filter_img, h_s, shift_by=5, box_h=100, box_w=50, threshold=10):

    [h, w] = np.shape(filter_img)

    # define box height and width
    box = [h_s, h_s + box_h, 0, box_w]

    # shift window by box_width/2
    shift_w = math.ceil(box_w/shift_by)
    shift_h = math.ceil(box_h/shift_by)
    # define num of horizontal shifts
    shift_horizontal = math.ceil((w - box_w)/shift_w)
    shift_vertical = math.ceil((h - (box_h + h_s))/shift_h)

    candidates_cord = []
    candidates_val = []
    # shift vertical
    for i in range(shift_vertical):

        for j in range(shift_horizontal):

            candidates_cord.append(list(box))
            #print(candidates_cord)
            # cut image
            #print(box)
            box_img = filter_img[int(box[0]):int(box[1]), int(box[2]):int(box[3])]
            box_sum = sum(map(sum, box_img))
            # print(box_sum)
            # save val
            candidates_val.append(box_sum)

            # shift horizontally
            box[2:4] = [x + shift_w for x in box[2:4]]

        # shift vertically
        box[:2] = [x + shift_h for x in box[:2]]
        box[2:4] = [0, box_w]

    #print(candidates_val)
    #print(candidates_cord)
    # get final candidates
    final_candidates = []
    for idx, val in enumerate(candidates_val):

        if val < threshold:
            #print(val)
            #print(candidates_cord[idx])
            final_candidates.append(candidates_cord[idx])

    return final_candidates


def find_candidates(filter_img, h_dots, w_dots, h_s, w_s):

    print(h_dots)
    print(len(h_dots))
    print(w_dots)
    print(len(w_dots))

    sum_lines = np.sum(filter_img, axis=1)
    sum_lines = np.multiply(sum_lines, sum_lines)

    #fig3 = plt.figure(3)
    #plt.plot(sum_lines)
    #plt.title('row')

    # sum of each col
    sum_col = np.sum(filter_img, axis=0)
    sum_col = np.multiply(sum_col, sum_col)

    # plot
    #fig4 = plt.figure(4)
    #plt.plot(sum_col)
    #plt.title('col')


    # find candidates using sliding window
    #vertical_boxes, h = get_candidatebox(filter_img, w_dots, h_s=h_s, w_s=w_s)
    #print('_______')
    #print(vertical_boxes)

    # 30, 3, 60, 10
    candidates = get_bounding_box(filter_img, h_s=50, shift_by=3, box_h=60, box_w=60, threshold=10)
    print('len = {}'.format(len(candidates)))
    #print(len(candidates))

    return candidates


# check h dots and w dots
    # note the changes
    #w_candidates = [0]
    #w_index = idx_change(w_candidates, sum_col, w_dots, threshold=1000)
    #h_candidates = [0]
    #h_index = idx_change(h_candidates, sum_lines, h_dots, threshold=800)

    #print(w_index)
    #print(h_index)





if __name__=='__main__':

    content_file = 'Test/test.txt'

    with open(content_file, 'r') as f:
        imgs = f.readlines()
        # remove \n from all list elements
        imgs = list(map(lambda s: s.strip(), imgs))


    for i in [4,6,10,28,33,39,46,47,56,71,88,116,128,140,144,154,186,193]:
        # select image
        # 1, 10, 28, 29, 39, 187
        print(i)
        test_img = imgs[i]
        print(test_img)

        # read img
        #img = mpimg.imread(test_img)
        img = cv2.imread(test_img)
        # change color space to YCrCb, digital version of yuv
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        #import sys
        #np.set_printoptions(threshold=sys.maxsize)
        #print(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)

        # compute horizontal scanlines
        #plot_horizontal_scanlines(img)

        #plot_vertical_scanlines(img)

        B, G, R = cv2.split(img)
        #lower_limit = np.array([80, 110, 110])
        #upper_limit = np.array([100, 130, 130])
        #print(B)
        green_mask = cv2.inRange(B, 0, 110)
        green_mask = green_mask/255
        print(green_mask)
        #print(green_mask)


        # get sample mask
        mask, h_dots, w_dots, h_s, w_s = scan_mask(img, row_dist=10, col_dist=15)

        filter_img = np.multiply(mask, green_mask)

        cand = find_candidates(filter_img, h_dots, w_dots, h_s, w_s)

        #import sys
        #np.set_printoptions(threshold=sys.maxsize)
        #print(mask)

        plot = True
        # plot
        if plot:

            fig2 = plt.figure(1)
            plt.imshow(green_mask)

            [h, w] = np.shape(filter_img)
            print(cand)
            for c in cand:
                #print(c)
                # plot frame
                plt.plot([c[2], c[3]], [c[0], c[0]], 'ro-')
                plt.plot([c[3], c[3]], [c[0], c[1]], 'ro-')
                plt.plot([c[3], c[2]], [c[1], c[1]], 'ro-')
                plt.plot([c[2], c[2]], [c[1], c[0]], 'ro-')

            #plt.matshow(mask)

            #plt.matshow(filter_img)

            #fig1 = plt.figure(2)
            #plt.imshow(img_yuv)

            # plot ground truth box
            #plot_groud_truth(test_img)

            plt.show()


