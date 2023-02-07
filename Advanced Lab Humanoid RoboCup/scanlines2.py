import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
import time


def plot_groud_truth(test_img):

    """
    plot ground truth bounding box of test image
    :param test_img:
    """

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


def scan_mask(w, h, row_dist=8, col_dist=10):
    """
    create mask with each dot samples the color
    :param img: original image
    :param row_dist: const distances between each horizontal line of dots
    :param col_dist: const distances between each vertical line of dots
    :return: mask with dots == 1, everything else == 0
    """

    # create list of to be sampled row/width
    row_list = []
    # list of to be sampled col/height
    col_list = []

    row = 0
    col = 0
    if type(h) == int and type(w) == int and col_dist > 0 and row_dist > 0:
        while row < h:
            row_list.append(row)

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

    return mask, row_list, col_list


def combine_box(final_candidates, c1, c2, id1, id2, small_list):
    """
    if 2 boxes shares the same line combine them
    :param final_candidates: list of box candidates
    :param c1: box1
    :param c2: box2
    :param id1: index of box1
    :param id2: index of box2
    :param small_list: new box
    :return:
    """
    new_c1 = []
    for i in small_list:
        # get the coordinates of new height/width
        if c1[i] < c2[i]:
            new_c1.append(c1[i])
        else:
            new_c1.append(c2[i])

    # new box
    new_c1.append(c1[2])
    new_c1.append(c1[3])

    # delete box
    final_candidates.pop(id1)
    if id1 < id2:
        # shorten id by 1 since list is gets shorter because previous element gets deleted
        final_candidates.pop(id2-1)
    else:
        final_candidates.pop(id2)

    final_candidates.append(new_c1)

    return final_candidates


def get_bounding_box(filter_img, mask, h_s=10, shift_by=2.0, box_h=100, box_w=50, threshold=0.8):
    """

    :param filter_img: filtered image where field val == 1 and non field val == 0
    :param mask: orignal mask with dots
    :param h_s: start height, we assume image top is unimportant
    :param shift_by: shift the window by image_size/shift_by, smaller value -> larger shift
    :param box_h: height of bounding box
    :param box_w: width of bounding box
    :param threshold: if sum within bounding box < threshold, keep box
    :return: list of [h1, h2, w1, w1] candidates
    """

    """ initialization """
    [h, w] = np.shape(filter_img)

    # define box height and width
    box = [h_s, h_s + box_h, 0, box_w]

    # shift window
    shift_w = math.ceil(box_w/shift_by)
    shift_h = math.ceil(box_h/shift_by)
    # define num of horizontal shifts
    shift_horizontal = math.ceil((w - box_w)/shift_w)
    shift_vertical = math.ceil((h - (box_h + h_s))/shift_h)

    mask_box = mask[int(box[0]):int(box[1]), int(box[2]):int(box[3])]
    mask_sum = sum(map(sum, mask_box))

    candidates_cord = []
    candidates_val = []

    """ shift predefined window to find large black spots (=0) in the image """
    for i in range(shift_vertical):
        for j in range(shift_horizontal):

            candidates_cord.append(list(box))
            # cut image
            box_img = filter_img[int(box[0]):int(box[1]), int(box[2]):int(box[3])]
            box_sum = np.sum(box_img, axis=0)
            box_sum = np.sum(box_sum)
            # save val
            candidates_val.append(box_sum)

            # shift horizontally
            box[2:4] = [x + shift_w for x in box[2:4]]

        # shift vertically
        box[:2] = [x + shift_h for x in box[:2]]
        box[2:4] = [0, box_w]

    """ get final candidates """
    final_candidates = []
    for idx, val in enumerate(candidates_val):
        # if percentage of uncovered spots in the window is smaller than threshold
        if val/mask_sum < threshold:
            # add 1 coord to candidates
            if len(final_candidates) == 0:
                final_candidates.append(candidates_cord[idx])

            # combine boxes that overlap
            else:
                # create new bigger combined box
                new_list = 0
                c = candidates_cord[idx]
                for cand in final_candidates:

                    # candidates sorted from left ot right and from top to bottom
                    # if there is no intersection in 1 dimenension
                    if (cand[1] < c[0]) or (c[1] < cand[0]) or (cand[3] < c[2]) or (c[3] < cand[2]):

                        # current candidates not intersecting with candidates in list
                        # if no coord in final coord intersects with c -> new list
                        new_list = new_list + 1

                    else:
                        # create bigger box
                        # find upper and lower bound of height or weight
                        for i in [0, 2]:
                            if candidates_cord[idx][i] <= cand[i]:
                                cand[i] = candidates_cord[idx][i]
                        for i in [1, 3]:
                            if candidates_cord[idx][i] >= cand[i]:
                                cand[i] = candidates_cord[idx][i]

                # new list
                if new_list == len(final_candidates):
                    final_candidates.append(candidates_cord[idx])

    # combine box that share a line
    for id1, c1 in enumerate(final_candidates):
        for id2, c2 in enumerate(final_candidates):

            # if 3 coord are the same, 2 boxes have the same line
            neighbor = 0
            for i in range(4):
                if c1[i] == c2[i]:
                    neighbor = neighbor+1

            # combine boxes if condition fulfilled
            if neighbor == 3:
                if id1 != id2 and [(c1[0] != c2[0]) and (c1[1] != c2[1])] and [(c1[2] == c2[2]) and (c1[3] == c2[3])]:
                    final_candidates = combine_box(final_candidates, c1, c2, id1, id2, small_list=[0,1])

                elif id1 != id2 and [(c1[0] == c2[0]) and (c1[1] == c2[1])] and [(c1[2] != c2[2]) and (c1[3] != c2[3])]:
                    final_candidates = combine_box(final_candidates, c1, c2, id1, id2, small_list=[2,3])


    return final_candidates


def plot_sum_of_axis(filter_img):
    """

    :param filter_img:
    :return: plot of sum along x and y axis
    """

    sum_lines = np.sum(filter_img, axis=1)
    sum_lines = np.multiply(sum_lines, sum_lines)

    plt.figure(3)
    plt.plot(sum_lines)
    plt.title('sum of horizontal lines')

    # sum of each col
    sum_col = np.sum(filter_img, axis=0)
    sum_col = np.multiply(sum_col, sum_col)

    # plot
    plt.figure(4)
    plt.plot(sum_col)
    plt.title('sum of vertical lines')


def field_boundaries(filter_img, h_dots, w_dots):
    """
    uses linear fit to predict field boundary
    piece wise linear func not working that well
    HULKS already have field detection, this is for python only
    :param filter_img:
    :param h_dots: list of pos of height pixel
    :param w_dots: list of pos of width pixel
    :return: predicted field boundary
    """

    # get x and y values
    x = w_dots.copy()
    y = []

    # search along a vertical line from top to bottom
    # the height idx of first 1 is taken as y value
    outlier = []
    for idx, i in enumerate(w_dots):
        check = 0
        # find first 1 along each col
        for j in h_dots:

            if filter_img[j][i] == 1:
                y.append(j)
                check = 1
                break

        # if no 1 in the vertical line
        # remove corresponding x value
        if check == 0:
            outlier.append(idx)

    for i in sorted(outlier, reverse=True):
        del x[i]

    # for dots that are not in line with field boundary, we want to remove them
    outlier = []
    # normal distance between 2 horizontal lines
    dist_h = h_dots[2] - h_dots[1]

    for idx, i in enumerate(y):
    # better method to remove outliers needed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if idx < len(y) - 4:
            # if dist between 2 lines is larger than 3*normal height distance -> outlier
            if abs(i - y[idx+4]) > 3*dist_h and idx not in outlier:
                outlier.append(idx)

    # remove corresponding x values
    if len(outlier) != 0:
        for i in sorted(outlier, reverse=True):
            del x[i]
            del y[i]

    # piece wise linear fit, not working well
    """
    # piece wise linear func
    # find highest y value
    min_idx = np.argmin(y)

    x1 = x[:min_idx]
    x2 = x[min_idx:]
    y1 = y[:min_idx]
    y2 = y[min_idx:]
    #print(len(y1))
    #print(y)

    m = []
    # fit using linear func
    if len(x1) > 0:
        m.append(np.polyfit(x1, y1, 1))
    if len(x2) > 0:
        m.append(np.polyfit(x2, y2, 1))
    print(len(m))

    # predict boundaries
    new_y = []
    for idx, i in enumerate(w_dots):
        if len(m) == 1:
            new_y.append(math.floor(m[0][0]*i + m[0][1]) - 10)
        else:
            if idx < min_idx:
                new_y.append(math.floor(m[0][0]*i + m[0][1]) -10)
            else:
                new_y.append(math.floor(m[1][0]*i + m[1][1]) -10)
    """

    # 1d linear fit
    m = np.polyfit(x, y, 1)

    # predict boundaries
    new_y = []
    for i in w_dots:
        new_y.append(math.floor(m[0]*i + m[1]) - 10)

    return new_y


def fill_img(filter_i, field, h_dots, w_dots):
    """

    :param filter_i: current filtered image
    :param field: field boundary, list of height pos
    :param h_dots:
    :param w_dots:
    :return: filtered image with filled background above the field
    """

    for idx, i in enumerate(w_dots):
        for j in h_dots:

            if j < field[idx]:

                filter_i[j][i] = 1

    return filter_i


start_time = time.time()

if __name__=='__main__':

    content_file = 'Test/test.txt'

    with open(content_file, 'r') as f:
        imgs = f.readlines()
        # remove \n from all list elements
        imgs = list(map(lambda s: s.strip(), imgs))

    # test the following images from test folder
    for i in [4,6,10,28,33,39,46,47,56,71,88,116,128,140,144,154,186,193]:
    #for i in range(0, 1249):
        # select image 1, 6, 10, 28, 29, 39, 56, 187, 1089

        #print('image num = {}'.format(i))
        test_img = imgs[i]

        """ read img and split channel """
        img_bgr = cv2.imread(test_img)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

        # plot images in different channels
        """
        test channel
        # change color space to YCrCb, digital version of yuv
        img_ycrbr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        img_ycr_cb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)

        y, u, v = cv2.split(img_yuv)
        y1, cr1, br1 = cv2.split(img_ycrbr)
        y2, cr2, br2 = cv2.split(img_ycr_cb)
        h, l, s = cv2.split(img_hls)
        h1, s1, v1 = cv2.split(img_hsv)
        l1, a, b = cv2.split(img_lab)
        #print(np.shape(img_gray))
        
        #channel = [B, G, R, y, u, v, y1, cr1, br1, y2, cr2, br2, h, l, s, h1, s1, v1, l1, a, b, img_gray]
        #chan = ['B', 'G', 'R', 'y', 'u', 'v', 'y1', 'cr1', 'br1', 'y2', 'cr2', 'br2',
        #        'h', 'l', 's', 'h1', 's1', 'v1', 'l1', 'a', 'b', 'gray']

        channel = [B, R, y, y1, cr1, s, s1, a]
        chan = ['B', 'R', 'y', 'y1', 'cr1', 's', 's1', 'a']
        
        for id, i in enumerate(channel):
            plt.figure(id+1)
            plt.imshow(i)
            plt.title(chan[id])
        
        plt.show()
        """

        # split the channels
        B, G, R = cv2.split(img_bgr)
        #print(B)

        """ find best threshold """
        [h, w] = np.shape(B)    # 384, 512
        upper_avg = np.sum(B[0][:])/w
        lower_avg = np.sum(B[h-1][:])/w
        color_thresh = ((upper_avg-lower_avg)/2) + lower_avg
        #print('color threshold = {}'.format(color_thresh))

        # get binary image according to threshold
        green_mask = cv2.inRange(B, 0, color_thresh)
        green_mask = green_mask/255

        """ create mask of dots to get filtered image """
        mask, h_dots, w_dots = scan_mask(w, h, row_dist=8, col_dist=10)
        #print(w_dots)

        # get filtered image by element wise product of mask and green mask
        filter_img = np.multiply(mask, green_mask)

        # find field boundaries
        field = field_boundaries(filter_img, h_dots, w_dots)
        #print(field)

        # fill values that are outside the field
        filter_img = fill_img(filter_img, field, h_dots, w_dots)

        # test possible parameters
        """ get bounding box candidates """
        # good
        #cand = get_bounding_box(filter_img, mask, h_s=100, shift_by=1.0, box_h=60, box_w=60, threshold=0.6)
        # this would return better bounding boxes
        cand = get_bounding_box(filter_img, mask, h_s=10, shift_by=32, box_h=75, box_w=75, threshold=0.4)


        #cand = get_bounding_box(filter_img[100:][:], mask, h_s=50, shift_by=2, box_h=50, box_w=50, threshold=0.4)

        # detect far field robots
        #cand = get_bounding_box(filter_img[0:100][:], mask, h_s=0, shift_by=1.0, box_h=16, box_w=10, threshold=0.2)

        # this is good
        #cand = get_bounding_box(filter_img[:][:], mask, h_s=0, shift_by=1.0, box_h=30, box_w=30, threshold=0.3)

#        cand = get_bounding_box(filter_img, mask, h_s=20, shift_by=1.0, box_h=30, box_w=30, threshold=0.6)
        #cand = get_bounding_box(filter_img[10:150][:], mask, h_s=10, shift_by=1, box_h=70, box_w=70, threshold=0.7)
        #cand1 = get_bounding_box(filter_img[10:200][:], mask, h_s=10, shift_by=1, box_h=100, box_w=100, threshold=0.8)

        #print('num of candidates = {}'.format(len(cand)))

        """ plot images """
        plot = 0
        # plot
        if plot:

            #plt.figure(0)
            #plt.imshow(B)

            fig2 = plt.figure(1)
            plt.imshow(green_mask)      # plot mask

            #fig1 = plt.figure(1)       # plot yuv image
            #plt.imshow(img_yuv)

            # plot the sum of dots against height and width
            # plot_sum_of_axis(filter_img, h_dots, w_dots)

            # plot field line
            plt.plot(w_dots, field, 'b-')
            for c in cand:
                # plot frame
                plt.plot([c[2], c[3]], [c[0], c[0]], 'ro-')
                plt.plot([c[3], c[3]], [c[0], c[1]], 'ro-')
                plt.plot([c[3], c[2]], [c[1], c[1]], 'ro-')
                plt.plot([c[2], c[2]], [c[1], c[0]], 'ro-')

            # plot mask
            #plt.matshow(mask)
            #plt.matshow(filter_img)

            # plot orignal image in YUV
            #fig1 = plt.figure(2)
            #plt.imshow(img_yuv)

            # plot ground truth box
            #plot_groud_truth(test_img)

            plt.show()


print("--- %s seconds ---" % (time.time() - start_time))

