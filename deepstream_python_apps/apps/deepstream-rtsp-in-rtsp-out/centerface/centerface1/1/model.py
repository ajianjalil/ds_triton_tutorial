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

import numpy as np
import cv2
import cupy as cp


class MEM:
    def __init__(self):
      self.Counter=1
      self.IM=np.zeros((3,200,200))

    def ADD(self):
        self.Counter=self.Counter+1
    def Set(self,IM):
        self.IM=np.maximum(IM,self.IM)-10
        self.IM=np.clip(self.IM,0,255)
        IM2=np.zeros((200,200,3))
        IM2[:,:,0]=self.IM[2,:,:]
        IM2[:,:,1]=self.IM[1,:,:]
        IM2[:,:,2]=self.IM[0,:,:]
        cv2.imshow('A',IM2.astype('uint8'))
        cv2.waitKey(3)

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

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
 
    def initialize(self, args):
      self.MEM1=MEM()
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
        # logger.log_error(f"Info Msg!:::::::::{test_module.value}")
        # logger.log_warn("Warning Msg!")
        # logger.log_error("Error Msg!")
        # logger.log_verbose("Verbose Msg!")
        for request in requests:
            
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            frame_cp = cp.fromDlpack(input_tensor.to_dlpack())
            logger.log_warn(f"Warning Msg!{frame_cp.device}")
            # frame = input_tensor.as_numpy()
            frame = cp.asnumpy(frame_cp)
            out_tensor = pb_utils.Tensor.from_dlpack(
                "OUTPUT0", input_tensor.to_dlpack()
            )

            stats = np.array([
              [360, 780, 360, 360, -1]           # Middle rectangle
          ])
            Brightest=np.sum(frame[0,:,:,:],axis=0)
            X,Y=np.where(Brightest==np.max(Brightest))
            X=X[0]
            Y=Y[0]
            X=np.clip(X,100,frame.shape[2]-100)
            Y=np.clip(Y,100,frame.shape[3]-100)
            # logger.log_warn(f"Warning Msg:{(X,Y)}")
            # logger.log_warn(f"shape{frame.shape}")

            stats1 = np.array(
              [Y-100, X-100, 500, 500, -1]   )        # Middle rectangl

            Brightest=np.sum(frame[1,:,:,:],axis=2)
            X1,Y1=np.where(Brightest==np.max(Brightest))
            X1=X1[0]
            Y1=Y1[0]

            X1=np.clip(X1,100,frame.shape[2]-100)
            Y1=np.clip(Y1,100,frame.shape[3]-100)

            stats2 = np.array(
              [Y1-100, X1-100, 500, 500, -1] )          # Middle rectangl


            batch_size = frame.shape[0]
            replicated_array = np.tile(stats, (batch_size, 1, 1))

            #replicated_array=np.array([stats1,stats2])
            replicated_array=np.array([stats1])
            replicated_array = np.tile(replicated_array, (batch_size, 1, 1))

            logger.log_warn(f"Warning Msg:{replicated_array.shape}")
            stats = replicated_array.astype(np.float32)

            out_tensor_1 = pb_utils.Tensor(
                "OUTPUT1", stats
            )
            # self.MEM1.Set(frame[0,:,:200,:200])

            responses.append(pb_utils.InferenceResponse([out_tensor,out_tensor_1]))
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")