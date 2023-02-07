import numpy as np
import cv2

a = np.loadtxt('candidates.txt')
name = 'Test/ScreenShot0'

for i in range(a.shape[0]):

    str_num = int(a[i, 0])
    if str_num < 10:
        file_name = name + '000' + str(str_num) + '.png'
    elif str_num < 100:
        file_name = name + '00' + str(str_num) + '.png'
    elif str_num < 1000:
        file_name = name + '0' + str(str_num) + '.png'
    else:
        file_name = name + str(str_num) + '.png'
    img = cv2.imread(file_name)

    cropImg = img[int(a[i, 1]):int(a[i, 2]), int(a[i, 3]):int(a[i, 4])]
    cv2.imwrite('Cutted_img/cutRobot_' + str(i) + '.png', cropImg)


