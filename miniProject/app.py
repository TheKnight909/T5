import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO('miniProject/yolov8_trained.pt')  # Replace with your custom-trained model if needed

st.title("Motorcycle Helmet Detection")

# Upload file (image or video)
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Check if it's an image
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Perform detection on the image
        results = model(image_np)

        # Annotate the image with detections
        annotated_image = results[0].plot()

        # Display the annotated image
        st.image(annotated_image, caption="Processed Image", use_column_width=True)

        # Option to download the annotated image
        annotated_image_pil = Image.fromarray(annotated_image)
        img_download = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        annotated_image_pil.save(img_download.name)

        with open(img_download.name, "rb") as file:
            st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="processed_image.png",
                mime="image/png"
            )
        os.remove(img_download.name)

    # If it's a video
    elif uploaded_file.type in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Open the video file
        video = cv2.VideoCapture(tfile.name)

        # Get the video frame width, height, and FPS
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Create a temporary file to save the processed video
        processed_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

        # Define the codec and create a VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_file.name, fourcc, fps, (frame_width, frame_height))

        # Initialize tracker
        tracker = cv2.TrackerCSRT_create()

        # Variable to store tracking state
        tracking = False
        bbox = None

        # Process the video frame by frame
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            if not tracking:
                # Perform detection
                results = model(frame)
                
                # Extract bounding boxes from results
                boxes = results[0].boxes.xyxy.numpy()
                
                # Initialize tracking with the first detected object
                if len(boxes) > 0:
                    bbox = boxes[0]  # Take the first detected object
                    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
                    tracker.init(frame, bbox)
                    tracking = True
            else:
                # Update tracker
                success, bbox = tracker.update(frame)
                if success:
                    # Draw bounding box
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

            # Annotate the frame with detections and tracking
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video file
            out.write(annotated_frame)

        # Release video objects
        video.release()
        out.release()

        # Provide a download button for the processed video
        st.success("Video processing complete! Download your video below:")

        # Open the processed video file in binary mode for download
        with open(processed_video_file.name, "rb") as video_file:
            st.download_button(
                label="Download Processed Video",
                data=video_file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

        # Remove temporary files after download (optional cleanup)
        os.remove(tfile.name)
        os.remove(processed_video_file.name)
