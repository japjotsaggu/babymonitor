# Assignment: Baby monitor video feed object detection and description

The code is written within a Colaboratory notebook 'Captioning_and_Detection.ipynb'. This notebook has two main sections, corresponding to dinstinct tasks in the assignment. Below is a concise overview of each part, along with instructions on how to execute the code for any alternate video file:

- Open the notebook in colab.
- Change the hardware accelerator to T4 GPU.

## Description Task (Salesforce/blip-image-captioning-base)

- Upload the video to session storage.
<img width="232" alt="Screenshot 2024-04-03 at 4 27 22 PM" src="https://github.com/japjotsaggu/babymonitor/assets/119132799/36c19477-dd36-4869-93fc-763cdd8fc422">

- Run the subsequent cells. The current video stored at babymonitor/videoplayback.mp4 (also linked [here](https://youtu.be/fm6PLc9j3OE?si=vwDvOWqxqsLgWzHQ)), has already undergone processing with the model, and the frame-by-frame descriptions are presented as cell output.

## Object detection (YOLOv8)

- The cell output exhibits frame-wise detections for [this](https://youtu.be/fm6PLc9j3OE?si=vwDvOWqxqsLgWzHQ) video (uploaded in session storage as videoplayback.mp4). 

- To overlay bounding boxes onto the video feed, execute the following cells to generate the resulting video named 'bb.mp4'. The outcome of this session has been uploaded as 'result.mp4' in this repository.
   
