import cupy as cp
from cucim.skimage.filters import gabor_kernel
from skimage import io
from matplotlib import pyplot as plt 
import numpy as np
import cv2
import cupy
import cucim
from skimage.io import imread, imshow

def gpu_imshow(image_gpu):
    image = cp.asnumpy(image_gpu)
    cv2.imshow("window",image)
    cv2.waitKey(0)


class CudaArrayInterface:
    def __init__(self, gpu_mat):
        w, h = gpu_mat.size()
        type_map = {
            cv2.CV_8U: "u1", cv2.CV_8S: "i1",
            cv2.CV_16U: "u2", cv2.CV_16S: "i2",
            cv2.CV_32S: "i4",
            cv2.CV_32F: "f4", cv2.CV_64F: "f8",
        }
        self.__cuda_array_interface__ = {
            "version": 2,
            "shape": (h, w),
            "data": (gpu_mat.cudaPtr(), False),
            "typestr": type_map[gpu_mat.type()],
            "strides": (gpu_mat.step, gpu_mat.elemSize()),
        }


img = np.zeros(shape=(640,360),dtype=np.uint8)
img[100:200,100:200] = 255
img[400:450,100:200] = 255
img[500:550,150:250] = 255

threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
threshold = threshold.astype(np.uint8)

d_img = cv2.cuda_GpuMat()
d_labels = cv2.cuda_GpuMat()

d_img.upload(threshold)

cp_labels = cupy.asarray(CudaArrayInterface(d_labels))

# image = imread('https://idr.openmicroscopy.org/webclient/render_image_download/9844418/?format=tif')

# imshow(image)
image_cpu = cv2.imread("testImg.png")
image_gpu = cp.asarray(image_cpu)

print(image_gpu.shape)

single_channel_gpu = image_gpu[:,:,1]

# the following line would fail
# imshow(single_channel_gpu)

# get single channel image back from GPU memory and show it
single_channel = cp.asnumpy(single_channel_gpu)
imshow(single_channel)

from cucim.skimage.filters import gaussian

blurred_gpu = gaussian(single_channel_gpu, sigma=5)

gpu_imshow(blurred_gpu)

def impulse_response(r, c, sigma=1):
    return np.exp(-(r**2 + c**2) / (2 * sigma**2))

from cucim.skimage.filters import LPIFilter2D
filter_instance = LPIFilter2D(impulse_response)

filtered = cucim.skimage.filters.filter_forward(image_gpu[:,:,1],predefined_filter=filter_instance)
gpu_imshow(filtered)