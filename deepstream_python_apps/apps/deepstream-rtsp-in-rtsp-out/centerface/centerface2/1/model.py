# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import json
import cv2
import numpy as np

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


def convert(connected_components_output):
    # Convert the connected components output to the specified format
    # print(connected_components_output)
    stats = np.array([
        connected_components_output[:, 0] + 360,  # x
        connected_components_output[:, 1] + 360,  # y
        connected_components_output[:, 2],  # width
        connected_components_output[:, 3],  # height
    ]).T

    # Add an extra column for the last value in the specified format
    stats = np.column_stack((stats, np.zeros(stats.shape[0])))
    return stats
    # return connected_components_output


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    #     """`initialize` is called only once when the model is being loaded.
    #     Implementing `initialize` function is optional. This function allows
    #     the model to initialize any state associated with this model.

    #     Parameters
    #     ----------
    #     args : dict
    #       Both keys and values are strings. The dictionary keys and values are:
    #       * model_config: A JSON string containing the model configuration
    #       * model_instance_kind: A string containing model instance kind
    #       * model_instance_device_id: A string containing model instance device ID
    #       * model_repository: Model repository path
    #       * model_version: Model version
    #       * model_name: Model name
    #     """

    #     # You must parse model_config. JSON string is not parsed here
    #     self.model_config = model_config = json.loads(args["model_config"])

    #     # Get OUTPUT0 configuration
    #     output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_0")

    #     # Convert Triton types to numpy types
    #     self.output0_dtype = pb_utils.triton_string_to_numpy(
    #         output0_config["data_type"]
    #     )




    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []

        logger = pb_utils.Logger
        # logger.log_info("Info Msg!")
        # logger.log_warn("Warning Msg!")
        # logger.log_error("Error Msg!")
        # logger.log_verbose("Verbose Msg!")
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_tensor1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            rect = input_tensor1.as_numpy()

            logger.log_warn(f"{rect.shape}")
            # print(rect)
            frame = input_tensor.as_numpy()
            # logger.log_warn(f"{frame.shape}")
            batch_size = frame.shape[0]
            stats = np.array([
                [10, 10, 200, 200, 3], 
                [1700, 10, 200, 200, 3], 
                [10, 860, 200, 200, 50], 
                [1700, 860, 200, 200, 100]
            ])
            # replicated_array = np.tile(stats, (batch_size, 1, 1))
            # stats = replicated_array.astype(np.float32)
            # number_of_items = 4
            # print(stats)
            # shape = np.array([number_of_items])
            
            # logger.log_warn(f"{frame.shape}")

            # connocted components algorithm
            roi = frame[0][:, 360:720,360:720]
            roi_frame = cv2.cvtColor(roi[1,:,:], cv2.COLOR_GRAY2RGB)
            
            # logger.log_warn(f"{frame.shape}")
            # logger.log_warn(f"{roi[1,:,:].shape}")
            # logger.log_warn(f"{roi[0,0,0:2]}")
            threshold = cv2.threshold(roi[0,:,:],127,255,cv2.THRESH_BINARY)[1]
            threshold = threshold.astype(np.uint8)
            cv2.imshow('Video', threshold)
            cv2.waitKey(1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)
            
            stats = convert(stats)

            number_of_items = stats.shape[0]
            
            stats = np.tile(stats, (batch_size, 1,1))
            # logger.log_warn(f"{stats.shape}")
            stats = stats.astype(np.float32)
            shape = np.array([number_of_items,5])
            shape = np.tile(shape, (batch_size, 1, 1))
            # logger.log_warn(f"{stats.shape}")
            # logger.log_warn(f"{stats[0]}")
            # logger.log_warn(f"{shape.shape}")
            shape = shape.astype(np.float32)
            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT0", stats    # list of recatangles for each channels
            )
            out_tensor_1 = pb_utils.Tensor(
                "OUTPUT1", shape   # list of shapes of recatangles for each channels
            )
            responses.append(pb_utils.InferenceResponse([out_tensor_0,out_tensor_1]))
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")