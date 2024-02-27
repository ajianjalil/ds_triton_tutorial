import cv2
import numpy as np
import time 
import cupy
def color_labels(labels):
    colors = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for r in range(labels.shape[0]):
        for c in range(labels.shape[1]):
            colors[r, c] = [labels[r, c] * 131 % 255, labels[r, c] * 241 % 255, labels[r, c] * 251 % 255]
    return colors


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


if __name__ == "__main__":
    input_image = "testImg.png"
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    img = np.zeros(shape=(640,360),dtype=np.uint8)
    img[100:200,100:200] = 255
    img[400:450,100:200] = 255
    img[500:550,150:250] = 255

    if img is None:
        print(f"Could not read input image file: {input_image}")
        exit(1)


    threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
    threshold = threshold.astype(np.uint8)

    # T0=time.time()
    # for q in range(0,1000):

    #     A=cv2.connectedComponentsWithStats(threshold, cv2.CV_32S, -1)
    # print(['CPU',(time.time()-T0)])


    
    d_img = cv2.cuda_GpuMat()
    d_labels = cv2.cuda_GpuMat()

    d_img.upload(threshold)
    connectivity = 8
    ltype = cv2.CV_32S
    ccltype =cv2.CCL_DEFAULT # cv2.CCL_BOLELLI  # BKE is the connected components labeling algorithm

    d_labels=cv2.cuda.connectedComponentsWithAlgorithm(d_img, connectivity, ltype, ccltype, d_labels)
    cp_d_labels = cupy.asarray(CudaArrayInterface(d_labels))
    h_labels = d_labels.download()

    # Initialize bounding boxes array on host
    bounding_boxes_host = np.zeros((h_labels.shape[0]*h_labels.shape[1] + 1, 4), dtype=np.int32)

    # Create a cupy array for bounding boxes on GPU
    bounding_boxes_gpu = cupy.asarray(bounding_boxes_host)

    # Load the custom CUDA kernel
    kernel_code = """
    extern "C" __global__
    void extractBoundingBoxes(const unsigned int* labels, int numRows, int numCols, int* boundingBoxes) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid < numRows * numCols) {
            unsigned int label = labels[tid];
            atomicMin(&boundingBoxes[4 * label], tid % numCols);    // min x
            atomicMax(&boundingBoxes[4 * label + 2], tid % numCols); // max x
            atomicMin(&boundingBoxes[4 * label + 1], tid / numCols); // min y
            atomicMax(&boundingBoxes[4 * label + 3], tid / numCols); // max y
        }
    }
    
    """
    """
    int newNumber;
    status = nppiCompressMarkerLabelsUF_32u_C1IR(d_labels, numCols * sizeof(unsigned int),{numCols, numRows}, numRows * numCols, &newNumber, d_buffer);
    if (status != NPP_SUCCESS) {
        // Handle NPP error
    }
    """
    kernel = cupy.RawKernel(kernel_code, "extractBoundingBoxes")

    # Launch the custom CUDA kernel
    grid_size = (h_labels.size + 255) // 256
    block_size = 256

    kernel((grid_size,), (block_size,), (cp_d_labels, h_labels.shape[0], h_labels.shape[1], bounding_boxes_gpu))

    # Copy the bounding boxes back to host
    bounding_boxes_host = cupy.asnumpy(bounding_boxes_gpu)
    values = bounding_boxes_host.copy()
    # Print or use the bounding boxes as needed
    print("Bounding Boxes:", bounding_boxes_host)
    for i in range(0, values.shape[0]):
        cv2.rectangle(img, (values[i][0], values[i][1]), (values[i][2], values[i][3]), (255, 0, 0), 2)
    print(len(values))
    cv2.imshow('Animating Squares', img)
    cv2.waitKey(0)