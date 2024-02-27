import cv2
import numpy as np
import time

# Create two matrices on the GPU
rows, cols = 1000, 1000
mat1 = cv2.cuda_GpuMat(rows, cols, cv2.CV_32F)
mat2 = cv2.cuda_GpuMat(rows, cols, cv2.CV_32F)

# Generate random data on the CPU and upload it to the GPU
mat1_cpu = np.random.randn(rows, cols).astype(np.float32)
mat2_cpu = np.random.randn(rows, cols).astype(np.float32)

mat1.upload(mat1_cpu)
mat2.upload(mat2_cpu)

# Create a destination matrix on the GPU
dst = cv2.cuda_GpuMat(rows, cols, cv2.CV_32F)

# Measure time for GPU matrix addition
start_time = time.time()
cv2.cuda.add(mat1, mat2, dst)
gpu_time = time.time() - start_time

# Download the result matrix from GPU to CPU
result_cpu = dst.download()

# Perform the same matrix addition on the CPU for comparison
result_cpu_reference = cv2.add(mat1_cpu, mat2_cpu)

# Compare the results
print("GPU Time:", gpu_time)

# Check if the GPU result matches the CPU result
if np.allclose(result_cpu, result_cpu_reference):
    print("Results Match!")
else:
    print("Results Do Not Match!")
