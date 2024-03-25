# ds_triton_tutorial
This will help us to learn how to use triton example from nvidia
# Explain the Problem
* Regarding Batch Order in Triton Inference Server with Python Backend:
We're developing a high-performance video analytics system using DeepStream with Triton Inference Server and a Python backend. Our requirement is to maintain a fixed order of channels within a batch. For example, camera 1 should correspond to batch[0], camera 2 to batch[1], and so on. However, we're encountering issues where the batch length varies randomly and the order changes. What solutions or configurations can ensure a consistent batch order?

* Mitigating Video Stuttering with Long Inference Times:
In our system, the execute method in the model.py takes up to 100ms to process a single frame. This leads to stuttering in the displayed video. Are there mechanisms, such as asynchronous inference or asynchronous overlay drawing using a buffer, that can help process all frames while avoiding stuttering?