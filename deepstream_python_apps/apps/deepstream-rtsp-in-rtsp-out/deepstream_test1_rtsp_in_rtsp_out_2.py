#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import sys
sys.path.append("../")
from common.bus_call import bus_call
from common.is_aarch_64 import is_aarch64
from common.FPS import PERF_DATA
import pyds
import platform
import math
import time
from ctypes import *
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib
import configparser
import datetime

import argparse

from ssd_parser import nvds_infer_parse_custom_tf_ssd, DetectionParam, NmsParam, BoxSizeParam


perf_data = None
MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]

# pgie_src_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.

CLASS_NB = 91
ACCURACY_ALL_CLASS = 0.5
UNTRACKED_OBJECT_ID = 0xffffffffffffffff
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
MIN_BOX_WIDTH = 32
MIN_BOX_HEIGHT = 32
TOP_K = 20
IOU_THRESHOLD = 0.3
OUTPUT_VIDEO_NAME = "./out.mp4"
MUXER_BATCH_TIMEOUT_USEC = 33000


def make_element(element_name, i):
    """
    Creates a Gstreamer element with unique name
    Unique name is created by adding element type and index e.g. `element_name-i`
    Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
    :param element_name: The name of the element to create
    :param i: the index of the element in the pipeline
    :return: A Gst.Element object
    """
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element


def get_label_names_from_file(filepath):
    """ Read a label file and convert it to string list """
    labels = "connected components"
    return labels


def make_elm_or_print_err(factoryname, name, printedname, detail=""):
    """ Creates an element with Gst Element Factory make.
        Return the element  if successfully created, otherwise print
        to stderr and return None.
    """
    print("Creating", printedname)
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        sys.stderr.write("Unable to create " + printedname + " \n")
        if detail:
            sys.stderr.write(detail)
    return elm


def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    # Intiallizing object counter with 0.
    obj_counter = dict(enumerate([0] * CLASS_NB))
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        id_dict = {
            val: index
            for index, val in enumerate(get_label_names_from_file("labels.txt"))
        }
        disp_string = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}"
        py_nvosd_text_params.display_text = disp_string.format(
            frame_number,
            num_rects,
            obj_counter[id_dict["car"]],
            obj_counter[id_dict["person"]],
        )

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def add_obj_meta_to_frame(frame_object, batch_meta, frame_meta, label_names):
    """ Inserts an object into the metadata """
    # this is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution.
    rect_params = obj_meta.rect_params
    rect_params.left = int(frame_object.left)
    rect_params.top = int(frame_object.top)
    rect_params.width = int(frame_object.width)
    rect_params.height = int(frame_object.height)

    # print(rect_params.height)

    # Semi-transparent yellow backgroud
    rect_params.has_bg_color = 0
    rect_params.bg_color.set(1, 1, 0, 0.4)

    # Red border of width 3
    rect_params.border_width = 3
    rect_params.border_color.set(1, 0, 0, 1)

    # Set object info including class, detection confidence, etc.
    obj_meta.confidence = frame_object.detectionConfidence
    obj_meta.class_id = frame_object.classId

    # There is no tracking ID upon detection. The tracker will
    # assign an ID.
    obj_meta.object_id = UNTRACKED_OBJECT_ID

    lbl_id = frame_object.classId
    if lbl_id >= len(label_names):
        lbl_id = 0

    # Set the object classification label.
    obj_meta.obj_label = label_names[lbl_id]

    # Set display text for the object.
    txt_params = obj_meta.text_params
    if txt_params.display_text:
        pyds.free_buffer(txt_params.display_text)

    txt_params.x_offset = int(rect_params.left)
    txt_params.y_offset = max(0, int(rect_params.top) - 10)
    txt_params.display_text = (
        label_names[lbl_id] + " " + "{:04.3f}".format(frame_object.detectionConfidence)
    )
    # Font , font-color and font-size
    txt_params.font_params.font_name = "Serif"
    txt_params.font_params.font_size = 10
    # set(red, green, blue, alpha); set to White
    txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    # Text background color
    txt_params.set_bg_clr = 1
    # set(red, green, blue, alpha); set to Black
    txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

    # Inser the object into current frame meta
    # This object has no parent
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)




def pgie_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    detection_params = DetectionParam(CLASS_NB, ACCURACY_ALL_CLASS)
    box_size_param = BoxSizeParam(IMAGE_HEIGHT, IMAGE_WIDTH,
                                  MIN_BOX_WIDTH, MIN_BOX_HEIGHT)
    nms_param = NmsParam(TOP_K, IOU_THRESHOLD)

    label_names = get_label_names_from_file("labels.txt")

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if (
                    user_meta.base_meta.meta_type
                    != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
            ):
                continue

            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

            # Boxes in the tensor meta should be in network resolution which is
            # found in tensor_meta.network_info. Use this info to scale boxes to
            # the input frame resolution.
            layers_info = []

            for i in range(tensor_meta.num_output_layers):
                layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                layers_info.append(layer)

            frame_object_list = nvds_infer_parse_custom_tf_ssd(
                layers_info, detection_params, box_size_param, nms_param
            )
            try:
                l_user = l_user.next
            except StopIteration:
                break

            for frame_object in frame_object_list:
                add_obj_meta_to_frame(frame_object, batch_meta, frame_meta, label_names)

        try:
            # indicate inference is performed on the frame
            frame_meta.bInferDone = True
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if ts_from_rtsp:
        if name.find("source") != -1:
            pyds.configure_source_for_ntp_sync(hash(Object))


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target(
            "src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):

    # Check input arguments
    number_sources = len(args)
    print(f"number of sources = {number_sources}")
    # global perf_data
    # perf_data = PERF_DATA(number_sources)
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")


    pipeline.add(streammux)

    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)


    queue1 = Gst.ElementFactory.make("queue", "queue1")
    
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", 4000000)

    print("Creating Pgie \n ")
    if gie=="nvinfer":
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    if gie=="nvinfer":
        pgie.set_property("config-file-path", "dstest1_pgie_config.txt")
    else:
        pgie.set_property("config-file-path", "dstest1_pgie_inferserver_config.txt")

    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        print(f"number_sources={number_sources}")
        pgie.set_property("batch-size", number_sources)

    print("Creating nvstreamdemux \n ")
    nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
    if not nvstreamdemux:
        sys.stderr.write(" Unable to create nvstreamdemux \n")

    

    
    pipeline.add(queue1)
    pipeline.add(pgie)
    pipeline.add(nvstreamdemux)


    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(nvstreamdemux)
    ##creating demux src

    for i in range(number_sources):
        # pipeline nvstreamdemux -> queue -> nvvidconv -> nvosd -> (if Jetson) nvegltransform -> nveglgl
        # Creating EGLsink
        if is_aarch64():
            print("Creating nv3dsink \n")
            sink = make_element("nv3dsink", i)
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = make_element("fakesink", i)
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")
        pipeline.add(sink)

        # creating queue
        queue = make_element("queue", i)
        pipeline.add(queue)

        # creating nvvidconv
        nvvideoconvert = make_element("nvvideoconvert", i)
        pipeline.add(nvvideoconvert)

        # creating nvosd
        nvdsosd = make_element("nvdsosd", i)
        pipeline.add(nvdsosd)
        nvdsosd.set_property("process-mode", OSD_PROCESS_MODE)
        nvdsosd.set_property("display-text", OSD_DISPLAY_TEXT)

        # connect nvstreamdemux -> queue
        padname = "src_%u" % i
        demuxsrcpad = nvstreamdemux.get_request_pad(padname)
        if not demuxsrcpad:
            sys.stderr.write("Unable to create demux src pad \n")

        queuesinkpad = queue.get_static_pad("sink")
        if not queuesinkpad:
            sys.stderr.write("Unable to create queue sink pad \n")
        demuxsrcpad.link(queuesinkpad)


        # connect  queue -> nvvidconv -> nvosd -> nveglgl
        queue.link(nvvideoconvert)
        nvvideoconvert.link(nvdsosd)


        nvvidconv_postosd = Gst.ElementFactory.make(
            "nvvideoconvert", f"convertor_postosd{i}")
        if not nvvidconv_postosd:
            sys.stderr.write(" Unable to create nvvidconv_postosd \n")

        # Create a caps filter
        caps = Gst.ElementFactory.make("capsfilter", f"filter{i}")
        caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        )

        encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder{i}")
        print("Creating H264 Encoder")

        if not encoder:
            sys.stderr.write(" Unable to create encoder")

        if is_aarch64():
            encoder.set_property("preset-level", 1)
            encoder.set_property("insert-sps-pps", 1)
            #encoder.set_property("bufapi-version", 1)

        sink = Gst.ElementFactory.make("rtspclientsink", f"rtspclientsink{i}")
        if not sink:
            sys.stderr.write(" Unable to create udpsink")
        sink.set_property("location",f"rtsp://127.0.0.1:554/video{i}")
        sink.set_property("protocols","tcp")
        print(f"will stream in rtsp://127.0.0.1:554/video{i}")
        pipeline.add(nvvidconv_postosd)
        pipeline.add(caps)
        pipeline.add(encoder)
        pipeline.add(sink)
        nvdsosd.link(nvvidconv_postosd)
        nvvidconv_postosd.link(caps)
        caps.link(encoder)
        encoder.link(sink)

        # sink.set_property("qos", 0)   ! nvvideoconvert ! nvv4l2h264enc bitrate=1500000 force-intra=false force-idr=false iframeinterval=30 profile=0 control-rate=1 ! rtspclientsink name=rtspclientsink location=rtsp://127.0.0.1:554/video1 protocols=tcp

    print("Linking elements in the Pipeline \n")
    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        # GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    # for i, source in enumerate(input_sources):
    #     print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                  help="Path to input H264 elementry stream", nargs="+", default=["a"], required=True)
    parser.add_argument("-g", "--gie", default="nvinfer",
                  help="choose GPU inference engine type nvinfer or nvinferserver , default=nvinfer", choices=['nvinfer','nvinferserver'])
    parser.add_argument("-c", "--codec", default="H264",
                  help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                  help="Set the encoding bitrate ", type=int)
    parser.add_argument("--rtsp-ts", action="store_true", default=False, dest='rtsp_ts', help="Attach NTP timestamp from RTSP source",
    )
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    global gie
    global ts_from_rtsp
    gie = args.gie
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    ts_from_rtsp = args.rtsp_ts
    print(f"Stream path is {stream_path}")
    stream_path = ['rtsp://192.168.8.33:8555/video1','rtsp://192.168.8.33:8555/video1',
                   'rtsp://192.168.8.33:8555/video1']
    stream_path = ['file:///opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/2.mp4','rtsp://192.168.8.33:8555/video1']
    stream_path = stream_path*2
    # stream_path = ['file:///opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/2.mp4','file:///opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/2.mp4',
    #                'file:///opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/2.mp4','file:///opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/2.mp4']
    return stream_path

if __name__ == '__main__':
    stream_path = parse_args()
    sys.exit(main(stream_path))
