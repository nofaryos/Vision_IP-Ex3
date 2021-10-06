# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import List
import numpy as np
import cv2


def opticalFlow_(im1: np.ndarray, im2: np.ndarray, step_size=15, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """

    # normalize
    im1 = im1 / 255
    im2 = im2 / 255

    rows_1, cols_1 = im1.shape[0], im1.shape[1]

    kernelX = np.array([[1, 0, -1]])
    kernelY = np.array([[1], [0], [-1]])

    conv2X = cv2.filter2D(im1, -1, kernelX)
    conv2Y = cv2.filter2D(im1, -1, kernelY)
    imageTime = im1 - im2

    pixels = []
    uvSol = []

    windowSize = int(win_size / 2)  # window_size

    for r in range(windowSize + step_size, rows_1 - windowSize, step_size):
        for c in range(windowSize + step_size, cols_1 - windowSize, step_size):
            pixels.append([c, r])
            #  The x derivative of im2
            Ix = conv2X[r - windowSize:r + windowSize + 1, c - windowSize:c + windowSize + 1].flatten()
            #  The y derivative of image im2
            Iy = conv2Y[r - windowSize:r + windowSize + 1, c - windowSize:c + windowSize + 1].flatten()
            #  The image difference im2 ‐ im1
            It = imageTime[r - windowSize:r + windowSize + 1, c - windowSize:c + windowSize + 1].flatten()

            a1 = np.sum(np.matmul(Ix, Ix))
            a2 = np.sum(np.matmul(Ix, Iy))
            a3 = np.sum(np.matmul(Iy, Ix))
            a4 = np.sum(np.matmul(Iy, Iy))

            sol = np.linalg.pinv(np.array([[a1, a2], [a3, a4]]))

            d1 = np.sum(np.matmul(Ix, It))
            d2 = np.sum(np.matmul(Iy, It))

            b = np.array([[-d1], [-d2]])
            u_v = np.matmul(sol, b)

            uvSol.append([u_v[0][0], u_v[1][0]])

    return np.array(pixels), np.array(uvSol)


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    # resize the image
    img = resizeImage(img, levels)
    pyr_lst = [img]
    # create gaussian kernel
    k_size = 5
    gau_k = cv2.getGaussianKernel(ksize=k_size, sigma=0.3 * ((k_size - 1) * 0.5 - 1) + 0.8)
    for i in range(1, levels):
        # filter the image
        I_tmp = cv2.filter2D(pyr_lst[i - 1], -1, gau_k)
        # down sample the image
        I_tmp = I_tmp[::2, ::2]
        pyr_lst.append(I_tmp)
    return pyr_lst


def resizeImage(img: np.ndarray, levels):
    """
    resize the image so the dimensions of the image will be divided by two
    :param img: Original image
    :param levels: Pyramid depth
    :return: resize image
    """
    rows, cols = img.shape[0], img.shape[1]
    rows = rows % pow(2, levels)
    cols = cols % pow(2, levels)

    if rows != 0:
        img = img[:-rows, :]

    if cols != 0:
        img = img[:, :-cols]
    return img


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """

    gray = isGrey(img)
    # gray image
    if gray:
        rows, cols = img.shape
        # zero padding
        padding_img = np.zeros((rows * 2, cols * 2)).astype(float)
        for r in range(rows):
            for c in range(cols):
                padding_img[2 * r][2 * c] = img[r][c]

    # color image
    else:
        rows, cols, dimension = img.shape
        # zero padding
        padding_img = np.zeros((rows * 2, cols * 2, 3)).astype(float)
        for r in range(rows):
            for c in range(cols):
                for d in range(dimension):
                    padding_img[2 * r][2 * c][d] = img[r][c][d]

    # blurring the image
    blur_image = cv2.filter2D(padding_img, -1, 4 * gs_k)

    return blur_image


def isGrey(img: np.ndarray):
    """
    Function that check if image is GRAY_SCALE image or RGB picture
    by checking the shape of the image
    :param img
    :ret
"""
    if len(img.shape) == 2:
        return True
    else:
        return False


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    # create gaussian pyramid
    gaussianPyramid = gaussianPyr(img, levels)

    # insert the last image in the gaussian pyramid to the laplacian pyramid
    laplacianPyramid = [gaussianPyramid[levels - 1]]

    # create gaussian kernel
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel = kernel.dot(kernel.T)

    # create the laplacian pyramid
    for i in reversed(range(1, levels)):
        # expand image
        expand = gaussExpand(gaussianPyramid[i], kernel)
        # reduce image
        reduce = gaussianPyramid[i - 1] - expand
        laplacianPyramid.insert(0, reduce)
    return laplacianPyramid


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)

    # create gaussian kernel
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel = kernel.dot(kernel.T)

    # the last image in laplacian pyramid = the last image in the gaussian pyramid
    origImage = lap_pyr[levels - 1]
    for i in reversed(range(levels - 1)):
        # expand image
        expandImage = gaussExpand(origImage, kernel)
        # connecting the image differences
        origImage = expandImage + lap_pyr[i]
    return origImage


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    # resize the images
    img_1 = resizeImage(img_1, levels)
    img_2 = resizeImage(img_2, levels)
    mask = resizeImage(mask, levels)

    # create laplacian pyramid for img_1
    lapPyrA = laplaceianReduce(img_1, levels)
    # create laplacian pyramid for img_2
    lapPyrB = laplaceianReduce(img_2, levels)
    # create gaussian pyramid for mask image
    gaussPyrM = gaussianPyr(mask, levels)
    lapPyrC = []

    # blend with pyramids
    for i in range(levels):
        lapPyrC.append(gaussPyrM[i] * lapPyrA[i] + (1 - gaussPyrM[i]) * lapPyrB[i])
    blended = laplaceianExpand(lapPyrC)

    # blend without pyramids
    naive = img_1 * mask + img_2 * (1 - mask)
    return naive, blended


def naive_blend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, isGrey: bool) -> (np.ndarray):
    naiveBlendImage = np.zeros_like(img_1)
    if isGrey:
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r][c] == 1:
                    naiveBlendImage[r][c] = img_1[r][c]
                else:
                    naiveBlendImage[r][c] = img_2[r][c]
    else:
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                for d in range(3):
                    if mask[r][c][1] == 1:
                        naiveBlendImage[r][c][d] = img_1[r][c][d]
                    else:
                        naiveBlendImage[r][c][d] = img_2[r][c][d]
    return naiveBlendImage
