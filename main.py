import cProfile
import pstats
from pstats import SortKey
import cv2
import numpy as np
def convolve(image, kernel):
    output = np.zeros((image.shape), image.dtype)
    for r in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] -1 ):
            value = 0
            for a in range(0, 3):
                for b in range(0, 3):
                    value = int(int(value) + int(kernel[a][b]) * int(image[r + a - 1, c + b - 1]))
            value = int(value / 8)
            if value < 0:
                value = 0
            if value > 255:
                value = 255
            output[r, c] = value
    return output


def basicGlobalThresholding(image):
    frequency = np.zeros(256, int)
#    print(frequency)

    # calculate the frequency
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            intensity = image[r][c]
            frequency[intensity] = frequency[intensity] + 1

    cumulativeFrequency = np.zeros(256, int)
    cumulativeFrequency[0] = frequency[0]
    for i in range(1, 256):
        cumulativeFrequency[i] = frequency[i] + cumulativeFrequency[i - 1]

    product = np.zeros(256, int)
    for i in range(0, 256):
        product[i] = frequency[i] * i

    cumulativeProduct = np.zeros(256, int)
    cumulativeProduct[0] = product[0]
    for i in range(1, 256):
        cumulativeProduct[i] = product[i] + cumulativeProduct[i - 1]

    iteration = 1
    T = int(cumulativeProduct[255] / cumulativeFrequency[255])
    while True:
        m1 = cumulativeProduct[T] / cumulativeFrequency[T]
        m2 = (cumulativeProduct[255] - cumulativeProduct[T]) / (cumulativeFrequency[255] - cumulativeFrequency[T])
        m = (m1 + m2) / 2
        print("Iteration: ", iteration, "T", T, "m1", m1, "m2", m2, "m", m)
        if T == int(m):
            break
        T = int(m)
        iteration = iteration + 1


    return T


def applyThreshold(image, T):
    output = np.zeros((image.shape), image.dtype)
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            if image[r, c] <= T:
                output[r, c] = 0
            else:
                output[r, c] = 255
    return int(output)

def getMagnitude(gx,gy):
    newgx = np.zeros((gx.shape),gx.dtype)
    newgy = np.zeros((gy.shape),gy.dtype)
    addition = np.zeros((gx.shape),gx.dtype)
    squre_root = np.zeros((gx.shape),gx.dtype)
    convert_integer = np.zeros((gx.shape),gx.dtype)
    newgx = np.multiply(gx,gx)
    newgy = np.multiply(gy,gy)
    addition = np.add(newgx,newgy)
    squre_root = np.sqrt(addition)
    convert_integer = squre_root.astype(int)
    for r in range(1, convert_integer.shape[0]):
        for c in range(1, convert_integer.shape[1]):
            if convert_integer[r , c] > 255:
                convert_integer[r , c] = 255


    return convert_integer

def get_direction(gx,gy):
    division = np.zeros((gx.shape),float)
    angle = np.zeros((gx.shape),float)
    angle_in_degrees = np.zeros((gx.shape),float)

    division = np.true_divide(gy,gx,where=(gx!=0) | (gy!=0))
    division[np.isnan(division) | (~np.isfinite(division))] = 0

    angle = np.arctan(division)
    angle_in_degrees = np.rad2deg(angle)
    return angle_in_degrees


def getSuppression(magnitude,direction):
    newImage = np.zeros((magnitude.shape), magnitude.dtype)
    height = magnitude.shape[0]
    weight = magnitude.shape[1]
    for r in range(0,magnitude.shape[0] -1 ):
        for c in range(0,magnitude.shape[1] -1):
            if direction[r,c] > -22.5 and direction[r,c] <=22.5 or direction[r,c] > 157.5 and direction[r,c] <= -157.5:
                if magnitude[r , c] < (magnitude[r -1 ,c] or magnitude[r + 1,  c]):
                    newImage[r , c] = 0
                else: newImage[r,c] = magnitude[r , c]
            elif direction[r,c] > 22.5 and direction[r,c] <= 67.5 or direction[r,c] > -157.5 and direction[r,c] <= -112.5:
                if magnitude[r,c] < (magnitude[r -1,c - 1] or magnitude[r + 1,c + 1]):
                    newImage[r,c] = 0
                else:newImage[r,c] = magnitude[r,c]
            elif direction[r,c] > 67.5 and direction[r,c] <= 112.5 or direction[r,c] > -112.5 and direction[r,c] <= -67.5:
                if magnitude[r ,c] < (magnitude [r , c -1] or magnitude[r ,c + 1]):
                    newImage[r , c] = 0
                else:newImage[r , c] = magnitude[r ,c]
            elif direction[r,c] > 112.5 and direction[r,c] <= 157.5 or direction[r,c] > -67.5 and direction[r,c] <= -22.5:
                if magnitude[r , c] < (magnitude[r - 1,c + 1] or magnitude[r + 1,c -1]):
                    newImage[r,c] = 0
                else:newImage[r ,c] = magnitude[r ,c]


    return newImage

def get_direction_radian(gx,gy):
    division = np.zeros((gx.shape),float)
    angle = np.zeros((gx.shape),float)
    angle_in_degrees = np.zeros((gx.shape),float)

    division = np.true_divide(gy,gx,where=(gx!=0) | (gy!=0))
    division[np.isnan(division) | (~np.isfinite(division))] = 0

    angle = np.arctan(division)
    return angle
def nonMaximaSuppression(magnitude,direction,th = 1):
    nms = np.zeros(magnitude.shape, magnitude.dtype)
    h,w = magnitude.shape
    for x in range(1, w-1):
        for y in range(1, h-1):
            mag = magnitude[y,x]
            if mag < th: continue
            teta = direction[y,x]
            dx, dy = 0, -1      # abs(orient) >= 1.1781, teta < -67.5 degrees and teta > 67.5 degrees
            if abs(teta) <= 0.3927: dx, dy = 1, 0       # -22.5 <= teta <= 22.5
            elif teta < 1.1781 and teta > 0.3927: dx, dy = 1, 1     # 22.5 < teta < 67.5 degrees
            elif teta > -1.1781 and teta < -0.3927: dx, dy = 1, -1  # -67.5 < teta < -22.5 degrees
            if mag > magnitude[y+dy,x+dx] and mag > magnitude[y-dy,x-dx]:
                nms[y,x] = mag
    return nms
def thresholds(image, tl, th):
    M,N = image.shape
    cf = {
        'WEAK': np.int32(50),
        'STRONG': np.int32(255),
    }

    strong_i, strong_j = np.where(image > th)
    weak_i, weak_j = np.where((image >= tl) & (image <= th))
    zero_i, zero_j = np.where(image < tl)

    image[strong_i, strong_j] = cf.get('STRONG')
    image[weak_i, weak_j] = cf.get('WEAK')
    image[zero_i, zero_j] = np.int32(0)
    weak = tl
    strong = 255
    for i in range(M):
        for j in range(N):
            if image[i, j] == weak:
                try:
                    if ((image[i + 1, j] == strong) or (image[i - 1, j] == strong)
                         or (image[i, j + 1] == strong) or (image[i, j - 1] == strong)
                         or (image[i+1, j + 1] == strong) or (image[i-1, j - 1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

image = cv2.imread("beetroot3.jpg")


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#canny = cv2.Canny(image, 70, 90)




kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
kernelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

def cannyEdgeDetector(image):
    pr = cProfile.Profile()
    pr.enable()
    kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    convolveKernelX = convolve(image,kernelX)
    convolveKernelY = convolve(image,kernelY)
    magnitude = getMagnitude(convolveKernelX,convolveKernelY)
    direction = get_direction(convolveKernelX,convolveKernelY)
    suppression = getSuppression(magnitude,direction)
    M = image.shape[0]
    N = image.shape[1]
    tl = 10
    th = 100
    weak = tl
    final_image = thresholds(suppression,tl,th)

    return final_image


output4 = cannyEdgeDetector(image)
#print(output4)
#cv2.imwrite("new.jpg",output4)

output = cv2.Canny(image,10,100)
cv2.imwrite("new1.jpg",output)
#cProfile.run('convolve(image,kernelX')
#cProfile.run('convolve,image,kernelY')


#Error diffrence between two image
errorRates = np.subtract(output,output4)
cv2.imwrite('Error_Rates.jpg',errorRates)
Sum = np.matrix(errorRates).sum()
print(sum)


cProfile.run('cannyEdgeDetector(image)','stats')
p = pstats.Stats('stats')
p.strip_dirs().sort_stats(-1).print_stats()
