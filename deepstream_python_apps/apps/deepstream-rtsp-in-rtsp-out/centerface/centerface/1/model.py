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
from tensorflow.experimental.dlpack import from_dlpack

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    # def initialize(self, args):
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

        # logger = pb_utils.Logger
        # logger.log_info("Info Msg!")
        # logger.log_warn("Warning Msg!")
        # logger.log_error("Error Msg!")
        # logger.log_verbose("Verbose Msg!")
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            tf_tensor = from_dlpack(input_tensor.to_dlpack())
            # print("shape={}".format(tf_tensor.shape))
            frame = tf_tensor.numpy()
            frame = np.squeeze(frame)
            frame = np.transpose(frame,(1,2,0))
            # print(frame[0, 1, 1, 1])
            # print(f"new shape={frame.shape}")
            threshold = cv2.threshold(frame[:,:,0],127,255,cv2.THRESH_BINARY)[1]
            threshold = threshold.astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)

            # Iterate through the connected components and print their statistics
            for i in range(1, num_labels):
                left, top, width, height, area = stats[i]
                # print(f"Component {i} - Left: {left}, Top: {top}, Width: {width}, Height: {height}, Area: {area}")
            out_tensor = pb_utils.Tensor(
                "OUTPUT0", stats
            )
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")