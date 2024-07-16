import numpy as np
from pyzbar.pyzbar import decode
import pyrealsense2 as rs
import cv2

def draw_rectangle(image, barcode):
    x, y, w, h = barcode.rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    barcode_data = barcode.data.decode('utf-8')
    print(barcode_data)
    cv2.putText(image, barcode_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def angle_between(p1, p2, p3):
    """Calculate the angle between the vectors (p1-p2) and (p3-p2) in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
    return np.abs(np.degrees(angle)) % 180



# Initialize the RealSense pipeline 848x480
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, framerate = 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, framerate = 30)

pipeline.start(config)

#create align object
align_to = rs.stream.color
align = rs.align(align_to)


try:
    frame_count = 0
    while True:
            try:
                
                # Wait for a new frame
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)
                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame :
                    continue

                
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.25), cv2.COLORMAP_JET)


                  # Define depth range for ROI (e.g., 0.5 to 1.5 meters)
                min_depth = 0  # meters
                max_depth = 0.8 # meters
                  # Create a mask based on the depth range
                mask = np.logical_and(depth_image > min_depth * 1000, depth_image < max_depth * 1000)

                # Convert mask to uint8
                mask = mask.astype(np.uint8) * 255
                masked_color_image = cv2.bitwise_and(color_image, color_image, mask=mask)

                gray = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)
                #blur = cv2.GaussianBlur(gray, (11, 11), 0)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
               
                edges = cv2.Canny(blur, 0, 200)

                # Apply morphological operations
                # Apply dilation
                K = 7
                kernel = np.ones((K, K), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=15)
                edges = cv2.erode(edges, kernel, iterations=15)
                
                                # Find contours
                contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Iterate through contours and find rectangles
                rectangles = []
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    
                    # Check if contour has 4 vertices (is rectangle)
                    if len(approx) == 4:
                
                        
                        angles = []
                        for i in range(4):
                            p1 = approx[i][0]
                            p2 = approx[(i + 1) % 4][0]
                            p3 = approx[(i + 2) % 4][0]
                            angle = angle_between(p1, p2, p3)
                            angles.append(angle)
                        
                        angle_min = np.min(np.abs(np.diff(angles)))

                        if all(70 < angle < 110 for angle in angles):
                            rectangles.append(approx)


                        # Calculate the center of the rectangle
                        M = cv2.moments(approx)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                        else:
                            center_x, center_y = 0, 0

                        height, width = masked_color_image.shape[:2]
                        d = np.sqrt((center_x-(width/2))**2 + (center_y-(height/2))**2)

                        print('height:', height, 'width', width)
                        print('center_x:', center_x, 'center_y:',center_y)
                        print(d)
                        # Draw the center
                        if d <=70:
                            cv2.circle(masked_color_image, (center_x, center_y), 5, (0, 0, 255), -1)
                            cv2.drawContours(masked_color_image, [approx], -1, (0, 255, 0),10)

                # Draw rectangles on original image
                #for rect in rectangles:
                  
        



                #barcodes = decode(color_image)
                #for barcode in barcodes:
                  #   draw_rectangle(color_image, barcode)
                scale_factor = 0.5
                resized_image = cv2.resize(masked_color_image,None,fx =  scale_factor,fy = scale_factor)
                resized_edges = cv2.resize(edges,None,fx =  scale_factor,fy = scale_factor)
                # Display the images
                cv2.imshow('Canny Image', resized_edges)
                cv2.imshow('Color Image', resized_image)
                #cv2.imshow('depth', depth_colormap)
        
               
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except RuntimeError as e:
                print("Error:", e)
                continue


finally:
    # Stop streaming
    pipeline.stop()
 
    cv2.destroyAllWindows()
    