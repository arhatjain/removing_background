# importing needed library
import cv2
import numpy as np


# load image
img1 = cv2.imread('D:\downloads/view1.jpeg')
img2 = cv2.imread('D:\downloads/view2.jpeg')
img3 = cv2.imread('D:\downloads/view3.jpeg')


# resizing image in(720*1080)format
image1=cv2.resize(img1, (1080,720))

image2=cv2.resize(img2, (1080,720))

image3=cv2.resize(img3, (1080,720))


# making original copy for the images
original1=image1.copy()
original2=image2.copy()
original3=image3.copy()


# converting image to grayscale
gray_image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray_image3=cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)


# function to remove back_ground
def background_removal(gray_image,original,canny_low,canny_high):
    cv2.imshow('input',original)
    cv2.waitKey(0)
    edges=cv2.Canny(gray_image,canny_low,canny_high)
    edges = cv2.dilate(edges,(2,2),iterations=5)
    edges = cv2.erode(edges,(2,2),iterations=3)
    cv2.imshow('edge_detected',edges)
    cv2.waitKey(0)
    im = cv2.GaussianBlur(edges, (0,0), sigmaX=0.5, sigmaY=0.5, borderType = cv2.BORDER_DEFAULT)

    ret,thresh=cv2.threshold(im, 127, 255,0)
    cv2.imshow('threshold',thresh)
    cv2.waitKey(0)
    
    contours,hirearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    min_area = 0.9*720*1080
    max_area = 0.95*720*1080
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    for contour in contours:
        if contour[0].all() > min_area and contour[0].all() < max_area:
        
            mask = cv2.fillConvexPoly(mask, contour[0], (255))
            mask = cv2.dilate(mask, None, iterations=15)
            mask = cv2.erode(mask, None, iterations=15)
    data = mask.tolist()

    for i in  range(len(data)):
        for j in  range(len(data[i])):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
        for j in  range(len(data[i])-1, -1, -1):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
    image = np.array(data)
    image[image !=  -1] =  255
    image[image ==  -1] =  0
    mask = np.array(image, np.uint8)

    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask ==  0] =  255
    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# calling function
background_removal(gray_image1, original1,400,900)
background_removal(gray_image2, original2,300,700)
background_removal(gray_image3, original3,350,800)
