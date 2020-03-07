#Watershed algorithm
#import the necessary packages
import cv2
import numpy
import matplotlib.pyplot as plt
import os
#load the color image 
sourcepath = 'multi_plant/'
outputpath = 'result_label/'
gtpath =  'multi_label/'
maskpath = 'mask_result/'
if not os.path.exists(sourcepath):
    print('Path' + sourcepath + 'does not exist')
    exit(1)
if not os.path.exists(outputpath):
    print('Path' + outputpath + 'does not exist')
    exit(1)
if not os.path.exists(gtpath):
    print('Path' + gtpath + 'does not exist')
    exit(1)
dices = []
a=0 #camera id
b=0 #plant id
c=0 #day id
d=0 #time id
for i in range (0, 900):

    imageName = f'rgb_0{a}_0{b}_00{c}_0{d}'
    labelName = f'label_0{a}_0{b}_00{c}_0{d}'
    d=d+1
    if d>5:
        d=0
        c=c+1
    if c>9:
        c=0
        b=b+1
    if b>4:
        b=0
        a=a+1     


    #read image and ground truth label
    img = cv2.imread(sourcepath + imageName + '.png' , cv2.IMREAD_COLOR)
    label = cv2.imread(gtpath + labelName + '.png', cv2.IMREAD_COLOR)
    
    #convert to hsv and inRange
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #range choosen manually with help of a colorpicker to highlight green color
    green = cv2.inRange(hsv, (20,37,30), (70,140,106))

    #noise removal
    kernel = numpy.ones((3,3), numpy.uint8)
    opening = cv2.morphologyEx(green,cv2.MORPH_OPEN,kernel)

    #sure background area
    sure_bg = cv2.dilate(opening,kernel, iterations = 8)

    #sure foreground area using Distance Transformation
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_C, 3)
    (ret, sure_fg) = cv2.threshold(dist_transform, 0.15*dist_transform.max(), 255, 0)

    #finding unknown region
    sure_fg = numpy.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    #marker labelling
    (ret, markers) = cv2.connectedComponents(sure_fg)

    #add one to all labels so that sure background is not 0 but 1
    markers = markers + 1

    #now mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    
    #saving to result folder
    plt.imshow(markers)
    plt.imsave(outputpath + imageName + '_segmented' + '.png', markers)

    #invert color of markers
    markers[markers==1] = 255
    markers= 255 - markers

    #saving mask result
    cv2.imwrite(maskpath + imageName + '_mask' + '.png', markers)
    

    #dice similarity
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    (thresh1, gt) = cv2.threshold(label, 1, 255, cv2.THRESH_BINARY)

    res = cv2.imread(maskpath + imageName + '_mask' + '.png')
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    (thresh2, seg) = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)


    k=255

    dice = numpy.sum(seg[gt==k])*2.0 / (numpy.sum(seg) + numpy.sum(gt))
    dices.append(dice)

    #end of for loop
print('mean dice sim score {}'.format(numpy.mean(dices)))