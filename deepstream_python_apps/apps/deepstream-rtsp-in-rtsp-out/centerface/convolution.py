import cv2
import numpy as np

# Reading the image
image = cv2.imread('testImg.png', cv2.IMREAD_COLOR)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert the image to grayscale (single-channel float32)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Define the kernel
kernel1 = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

# Upload the image and kernel to GPU
d_image = cv2.cuda_GpuMat()
d_image.upload(gray_image)

d_kernel = cv2.cuda_GpuMat()
d_kernel.upload(kernel1)

# Create a temporary GpuMat for the result
d_result = cv2.cuda_GpuMat()

# Create a CUDA-accelerated convolution filter
convolution_filter = cv2.cuda.createConvolution()

# Perform convolution
convolution_filter.convolve(d_image, d_kernel, result=d_result)

# Download the result from GPU
result = d_result.download()

# Check if the result image has valid dimensions
if result.shape[0] > 0 and result.shape[1] > 0:
    # Show the original and output image
    cv2.imshow('Original', gray_image)
    cv2.imshow('Convolved Image (CUDA)', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Convolution result has invalid dimensions.")
