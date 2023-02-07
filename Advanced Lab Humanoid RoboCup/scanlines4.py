# segmented sumary along height
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
        final_candidates.pop(id2 - 1)
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
    shift_w = math.ceil(box_w / shift_by)
    shift_h = math.ceil(box_h / shift_by)
    # define num of horizontal shifts
    shift_horizontal = math.ceil((w - box_w) / shift_w)
    shift_vertical = math.ceil((h - (box_h + h_s)) / shift_h)

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
    num_cand = 0
    for idx, val in enumerate(candidates_val):
        # if percentage of uncovered spots in the window is smaller than threshold
        if val / mask_sum < threshold:
            num_cand = num_cand + 1
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
                    neighbor = neighbor + 1

            # combine boxes if condition fulfilled
            if neighbor == 3:
                if id1 != id2 and [(c1[0] != c2[0]) and (c1[1] != c2[1])] and [(c1[2] == c2[2]) and (c1[3] == c2[3])]:
                    final_candidates = combine_box(final_candidates, c1, c2, id1, id2, small_list=[0, 1])

                elif id1 != id2 and [(c1[0] == c2[0]) and (c1[1] == c2[1])] and [(c1[2] != c2[2]) and (c1[3] != c2[3])]:
                    final_candidates = combine_box(final_candidates, c1, c2, id1, id2, small_list=[2, 3])

    return final_candidates, num_cand


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


def get_boundary_points(filter_img, h_dots, w_dots):
    """

    :param filter_img: scan img
    :param h_dots:
    :param w_dots:
    :return: list of dots at the field border
    """
    valid_x = []
    valid_y = []

    for h in h_dots:
        # sum of every horizontal line, if 0 boundary is still below
        if np.sum(filter_img[h][:]) == 0:
            valid_x = []
            valid_y = []

        else:
            for w in w_dots:
                if filter_img[h][w] != 0 and w not in valid_x:
                    # get highest line
                    valid_x.append(w)
                    valid_y.append(h)

    return valid_x, valid_y


def remove_outliers(x, y, k=3):
    """
    compares every point with its k neighbor, if y value differs too much, its an outlier
    :param x: x coord of border point candidates
    :param y: y coord of border point candidates
    :param k:
    :return: filtered candidate border points for border detection
    """
    # sort x and y
    x_idx = np.argsort(x)
    y = [y[i] for i in x_idx]
    x = sorted(x)

    for i in range(20):
        outlier = []
        for i in range(len(x)):
            if (k - 1) < i < len(x) - k:
                for j in range(1, k):
                    if (y[i] > y[i - j] and y[i] > y[i + j]):
                        if i not in outlier:
                            outlier.append(i)
        # print(outlier)
        if len(outlier) == 0:
            break
        else:
            for i in sorted(outlier, reverse=True):
                del x[i]
                del y[i]

    x = x[4:-4]
    y = y[4:-4]

    return x, y


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
    # get boundary points
    x, y = get_boundary_points(filter_img, h_dots, w_dots)
    # reduce points
    x, y = remove_outliers(x, y)

    # piece wise linear fit
    # divide into 2 sets
    x1 = []
    y1 = []
    for i in range(len(x)):
        if i < len(x) - 6:
            for j in range(6):
                if y[i] > y[i + j]:
                    if x[i] not in x1:
                        x1.append(x[i])
                        y1.append(y[i])
    x2 = []
    y2 = []
    for i in range(len(x)):
        if x[i] not in x1:
            x2.append(x[i])
            y2.append(y[i])

    m = []
    # fit using linear func
    if len(x1) > 0:
        m.append(np.polyfit(x1, y1, 1))
    if len(x2) > 0:
        m.append(np.polyfit(x2, y2, 1))

    # predict boundaries
    new_y = []
    for idx, i in enumerate(w_dots):
        if len(m) == 1:
            new_y.append(math.floor(m[0][0] * i + m[0][1]) - 5)
        else:
            if i < x2[0]:
                new_y.append(math.floor(m[0][0] * i + m[0][1]) - 5)
            else:
                new_y.append(math.floor(m[1][0] * i + m[1][1]) - 5)
    """

    # 1d linear fit
    m = np.polyfit(x, y, 1)

    # predict boundaries
    new_y = []
    for i in w_dots:
        new_y.append(math.floor(m[0] * i + m[1]) - 10)
    """
    return new_y, x, y


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


def save_as_txt(cand, i, start_element):
    """

    :param cand: bounding box (h1, h2, w1, w2)
    :param i: image index
    :param start_element:
    :return:
    """
    if i == start_element:
        # delete all previous texts
        open('candidates.txt', 'w').close()

    with open('candidates.txt', 'a') as f:

        for c in cand:
            f.write(str(i) + ' ' + str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + ' ' + str(c[3]) + '\n')


start_time = time.time()

if __name__ == '__main__':

    # txt file containing names of all imgs in test folder
    content_file = 'Test/test.txt'
    save_txt = 1
    plot = 1

    with open(content_file, 'r') as f:
        imgs = f.readlines()
        # remove \n from all list elements
        imgs = list(map(lambda s: s.strip(), imgs))

    # test the following images from test folder
    test_l = [4, 6, 10, 24, 28, 33, 39, 46, 47, 56, 71, 88, 111, 116, 128, 140, 144, 154, 186, 193]
    #test_l = [47]
    #test_l = range(0, 1249)
    for i in test_l:
        # select image 1, 6, 10, 28, 29, 39, 56, 144, 187, 1089

        # print('image num = {}'.format(i))
        test_img = imgs[i]

        """ read img and split channel """
        img_bgr = cv2.imread(test_img)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

        # split the channels
        B, G, R = cv2.split(img_bgr)
        # print(B)

        """ find best threshold """
        [h, w] = np.shape(B)  # 384, 512
        upper_avg = np.sum(B[0][:]) / w
        lower_avg = np.sum(B[h - 1][:]) / w
        color_thresh = ((upper_avg - lower_avg) / 2) + lower_avg
        # print('color threshold = {}'.format(color_thresh))

        # get binary image according to threshold
        green_mask = cv2.inRange(B, 0, color_thresh)
        green_mask = green_mask / 255

        """ create mask of dots to get filtered image """
        mask, h_dots, w_dots = scan_mask(w, h, row_dist=8, col_dist=10)
        # print(w_dots)

        # get filtered image by element wise product of mask and green mask
        filter_img = np.multiply(mask, green_mask)

        # find field boundaries
        field, x, y = field_boundaries(filter_img, h_dots, w_dots)
        # print(field)

        # fill values that are outside the field
        filter_img = fill_img(filter_img, field, h_dots, w_dots)

        # test possible parameters
        """ get bounding box candidates """

        # this would return pretty good bounding boxes
        # cand, cand_num = get_bounding_box(filter_img, mask, h_s=20, shift_by=3, box_h=75, box_w=75, threshold=0.5)

        # also pretty good - best
        cand, cand_num = get_bounding_box(filter_img, mask, h_s=20, shift_by=3, box_h=60, box_w=60, threshold=0.4)

        # detect far field robots - best (some smaller robots also detectable, but also more false alarm)
        # cand = get_bounding_box(filter_img, mask, h_s=20, shift_by=4, box_h=40, box_w=40, threshold=0.5)

        # cand = get_bounding_box(filter_img, mask, h_s=20, shift_by=3, box_h=30, box_w=30, threshold=0.3)

        """ far field detection - only detect between 0 to 100 """
        # create padding image
        # padded_img = np.ones([h, w])
        # padded_img[0:100][:] = filter_img[0:100][:]
        # cand = get_bounding_box(padded_img, mask, h_s=10, shift_by=4, box_h=30, box_w=30, threshold=0.4)

        #print('num of candidates before fusing = {}'.format(cand_num))

        """ save as txt """
        if save_txt:
            save_as_txt(cand, i, test_l[0])

        """ plot images """
        if plot:

            # plt.figure(0)
            # plt.imshow(B)

            # plot the sum of dots against height and width
            # plot_sum_of_axis(filter_img)

            fig2 = plt.figure(1)
            plt.imshow(green_mask)  # plot mask
            plt.title(i)

            # fig1 = plt.figure(1)       # plot yuv image
            # plt.imshow(img_yuv)

            # plot field line
            plt.plot(w_dots, field, 'r-')
            for c in cand:
                # plot frame
                plt.plot([c[2], c[3]], [c[0], c[0]], 'co-')
                plt.plot([c[3], c[3]], [c[0], c[1]], 'co-')
                plt.plot([c[3], c[2]], [c[1], c[1]], 'co-')
                plt.plot([c[2], c[2]], [c[1], c[0]], 'co-')

            # plot mask
            # plt.matshow(mask)
            #plt.matshow(filter_img)
            plt.plot(x, y, 'go')
            # plot field line
            # plt.plot(w_dots, field, 'b-')
            # plt.title(i)

            # plot orignal image in YUV
            # fig1 = plt.figure(2)
            # plt.imshow(img_yuv)

            # plot ground truth box
            # plot_groud_truth(test_img)

            plt.show()

print("--- %s seconds ---" % (time.time() - start_time))