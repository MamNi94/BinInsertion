import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color,  1920, 1080, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Define depth range for ROI (e.g., 0.5 to 1.5 meters)
        min_depth = 0  # meters
        max_depth = 0.7 # meters

        # Create a mask based on the depth range
        mask = np.logical_and(depth_image > min_depth * 1000, depth_image < max_depth * 1000)

        # Convert mask to uint8
        mask = mask.astype(np.uint8) * 255

        # Apply mask to the color image
        masked_color_image = cv2.bitwise_and(color_image, color_image, mask=mask)
        cv2.imshow('Masked Color Image', masked_color_image)
        # Convert to grayscale
        gray = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (11, 11), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Display the result
        cv2.imshow('Edges', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
