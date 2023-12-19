/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* This custom post processing parser is for centernet face detection model */
#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <tuple>
#include <memory>
#include <opencv2/opencv.hpp>

#define CLIP(a, min, max) (MAX(MIN(a, max), min))

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomTfSSD(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
										  NvDsInferNetworkInfo const &networkInfo,
										  NvDsInferParseDetectionParams const &detectionParams,
										  std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* This is a smaple bbox parsing function for the centernet face detection onnx model*/
struct FrcnnParams
{
	int inputHeight;
	int inputWidth;
	int outputClassSize;
	float visualizeThreshold;
	int postNmsTopN;
	int outputBboxSize;
	std::vector<float> classifierRegressorStd;
};

struct FaceInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float landmarks[10];
};

/* NMS for centernet */
static void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float nmsthreshold)
{
	std::sort(input.begin(), input.end(),
			  [](const FaceInfo &a, const FaceInfo &b) {
				  return a.score > b.score;
			  });

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;

		output.push_back(input[i]);

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1; //std::max(input[i].x1, input[j].x1);
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2; //bug fixed ,sorry
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score > nmsthreshold)
				merged[j] = 1;
		}
	}
}
/* For CenterNetFacedetection */
//extern "C"
static std::vector<int> getIds(float *heatmap, int h, int w, float thresh)
{
	std::vector<int> ids;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{

			//			std::cout<<"ids"<<heatmap[i*w+j]<<std::endl;
			if (heatmap[i * w + j] > thresh)
			{
				//				std::array<int, 2> id = { i,j };
				ids.push_back(i);
				ids.push_back(j);
				//	std::cout<<"print ids"<<i<<std::endl;
			}
		}
	}
	return ids;
}

/* customcenternetface */
extern "C" bool NvDsInferParseCustomCenterNetFace(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                                  NvDsInferNetworkInfo const &networkInfo,
                                                  NvDsInferParseDetectionParams const &detectionParams,
                                                  std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    // Clear the object list
    objectList.clear();

    // Create a constant face rectangle with fixed coordinates
    NvDsInferObjectDetectionInfo object;

    // Set fixed coordinates (200 * 200)
    object.left = 200;
    object.top = 200;
    object.width = 200;
    object.height = 200;

    // Clip object box coordinates to network resolution
    object.left = CLIP(object.left, 0, networkInfo.width - 1);
    object.top = CLIP(object.top, 0, networkInfo.height - 1);
    object.width = CLIP(object.width, 0, networkInfo.width - 1);
    object.height = CLIP(object.height, 0, networkInfo.height - 1);

    // Set other detection parameters
    object.detectionConfidence = 0.99;
    object.classId = 0;

    // Add the constant face rectangle to the object list
    objectList.push_back(object);

    return true;
}



/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCenterNetFace);
