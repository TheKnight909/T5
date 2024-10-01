import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8_Medium.pt')  # Ensure the model file is in the root directory of your Space

def run_yolo(image):
    # Run the model on the image and get results
    results = model(image)
    return results

def process_results(results, image):
    # Draw bounding boxes and labels on the image
    boxes = results[0].boxes  # Get boxes from results
    for box in boxes:
        # Get the box coordinates and label
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class index
        label = model.names[cls]  # Get class name from index
        
        # Draw rectangle and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

import tempfile

def process_video(uploaded_file):
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Get the path of the temporary file
    
    # Read the video file
    video = cv2.VideoCapture(temp_file_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    frames = []
    
    # Create a Streamlit progress bar, text for percentage, and timer
    progress_bar = st.progress(0)
    progress_text = st.empty()  # Placeholder for percentage text
    timer_text = st.empty()  # Placeholder for the timer
    
    current_frame = 0
    start_time = time.time()  # Start the timer
    
    while True:
        ret, frame = video.read()
        if not ret:
            break  # Break the loop if there are no frames left

        # Run YOLO model on the current frame
        results = run_yolo(frame)
        
        # Process the results and draw boxes on the current frame
        processed_frame = process_results(results, frame)
        frames.append(processed_frame)  # Save the processed frame
        
        current_frame += 1
        
        # Calculate and display the progress
        progress_percentage = (current_frame / total_frames) * 100
        progress_bar.progress(progress_percentage / 100)  # Update the progress bar
        progress_text.text(f'Processing: {progress_percentage:.2f}%')  # Update the percentage text

        # Calculate and display the elapsed time
        elapsed_time = time.time() - start_time
        timer_text.text(f'Elapsed Time: {elapsed_time:.2f} seconds')  # Update the timer text
    
    video.release()
    
    # Create a video writer to save the processed frames
    height, width, _ = frames[0].shape
    output_path = 'processed_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in frames:
        out.write(frame)  # Write each processed frame to the video

    out.release()
    
    # Complete the progress bar and show final message
    progress_bar.progress(100)
    progress_text.text('Processing: 100%')
    st.success('Video processing complete!')

    # Display the final elapsed time
    final_elapsed_time = time.time() - start_time
    timer_text.text(f'Total Elapsed Time: {final_elapsed_time:.2f} seconds')
    
    # Display the processed video
    st.video(output_path)

    # Create a download button for the processed video
    with open(output_path, 'rb') as f:
        video_bytes = f.read()
    st.download_button(label='Download Processed Video', data=video_bytes, file_name='processed_video.mp4', mime='video/mp4')

def main():
    st.title("Motorbike Violation Detection")

    # Upload file
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            # Process the image
            image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
            results = run_yolo(image)
            
            # Process the results and draw boxes on the image
            processed_image = process_results(results, image)
            
            # Display the processed image
            st.image(processed_image, caption='Detected Image', use_column_width=True)

        elif uploaded_file.type == "video/mp4":
            # Process the video
            process_video(uploaded_file)  # Process the video and save the output
            

if __name__ == "__main__":
    main()
