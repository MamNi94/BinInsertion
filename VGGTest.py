import numpy as np
import pyrealsense2 as rs
import cv2
import os

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



def cut_region_at_distance(depth_image,color_image,min_depth = 0,max_depth = 0.8, region = False):
                if region == False:
                    mask = np.logical_and(depth_image > min_depth * 1000, depth_image < max_depth * 1000)

                    # Convert mask to uint8
                    mask = mask.astype(np.uint8) * 255
                    masked_color_image = cv2.bitwise_and(color_image, color_image, mask=mask)
                    return masked_color_image

                if region == True:
                     
                    min_depth_mm = min_depth * 1000
                    max_depth_mm = max_depth * 1000

                    # Create a mask for the specified depth range
                    mask = np.logical_and(depth_image > min_depth_mm, depth_image < max_depth_mm)

                    # Convert mask to uint8
                    mask = mask.astype(np.uint8) * 255

                    # Find contours in the mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Create an empty mask to draw the contour region
                    contour_mask = np.zeros_like(mask)
                      # Check if any contours were found
                    if contours:
                        all_points = np.concatenate(contours)
                        hull = cv2.convexHull(all_points)
                        # Draw the filled contour on the contour_mask
                        cv2.drawContours(contour_mask, [hull], -1, (255), thickness=cv2.FILLED)

                    masked_color_image = cv2.bitwise_and(color_image, color_image, mask=contour_mask)

                    return masked_color_image

def detect_roi(binary_thresh,color_image, min_angle = 70, max_angle = 110, detect_center = True):
        box_detected = False
        
        center = [0,0]
      
        # Find contours
        contours, _ = cv2.findContours(binary_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Iterate through contours and find rectangles
        rectangles = []
       
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            bbox = approx
            # Check if contour has 4 vertices (is rectangle)
            if len(approx) == 4:
        
                angles = []
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % 4][0]
                    p3 = approx[(i + 2) % 4][0]
                    angle = angle_between(p1, p2, p3)
                    angles.append(angle)
                

                if all(min_angle < angle < max_angle for angle in angles):
                    rectangles.append(approx)
                    
                    if detect_center == True:
                       
                        # Calculate the center of the rectangle
                        M = cv2.moments(approx)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                        else:
                            center_x, center_y = 0, 0

                        height, width = color_image.shape[:2]
                        d = np.sqrt((center_x-(width/2))**2 + (center_y-(height/2))**2)

                    
                        # Draw the center
                        if d <=70:
                            cv2.circle(color_image, (center_x, center_y), 7, (0, 0, 255), 5)
                            cv2.putText(color_image,'Bin Detected', (center_x-20, center_y-50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 5 )
                            cv2.drawContours(color_image, [approx], -1, (0, 255, 0),10)
                            box_detected = True
                            center = [center_x,center_y]
                            bbox = approx
                            return  box_detected, center, color_image, bbox
                    
                    if detect_center == False:
                        
                        cv2.drawContours(color_image, [approx], -1, (0, 255, 255),10)
                        box_detected = True
                    
        return  box_detected, center, color_image, bbox
    
    
def detect_holes(x,y,w,h,image, number = 1):
    top_left_x = int(x - w / 2) 
    top_left_y = int(y - h / 2)
    bottom_right_x = int(x1 + w / 2)
    bottom_right_y = int(y1 + h / 2)
    #cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 255), 2)
                        
    # Ensure the coordinates are within the image boundaries
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(image.shape[1], top_left_x + w)
    bottom_right_y = min(image.shape[0], top_left_y + h)

    # Crop the image using slicing
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] 
    cv2.putText(cropped_image,f'x {x}', (0 ,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    cv2.putText(cropped_image,f'y {y}', (0 ,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    cv2.imshow(f'Cropped Corner {number}', cropped_image)
    
        
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust codec as needed (e.g., 'XVID', 'MJPG', 'MP4V')
out_color = cv2.VideoWriter('Demo/BinDetectionDay2_2.avi', fourcc,30.0, (1920, 1080))
# Check if the VideoWriter object is successfully initialized
if not out_color.isOpened():
    print("Error: Could not open video writer.")



#Initialize
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, framerate = 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, framerate = 30)

pipeline.start(config)

#create align object
align_to = rs.stream.color
align = rs.align(align_to)


try:
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

                #detect BOX
                masked_color_image = cut_region_at_distance(depth_image, color_image,min_depth = 0,max_depth = 0.8, region = True)
                gray = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)
                _, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                box_detected, center,color_image, box = detect_roi(binary_thresh,color_image, min_angle = 70, max_angle = 110,detect_center = True)
                print(box[:,0][:,0])
                print(box[:,0][:,1])
                print('....')
                if box_detected == True:
                    w, h = 224, 224 
                    x,y = box[:,0][:,0],box[:,0][:,1]
                    
                    x1 ,y1= max(box[:,0][:,0]) -250, min(box[:,0][:,1])
                    x2 ,y2= box[:,0][:,0][1], box[:,0][:,1][1]
                    
                    '''
                    i = 0
                    for p in x:
                        if x3 == p:
                            y3 = y[i]
                            x3 = x3+ 250
                            break
                        else:
                            i += 1  
                    '''
                    x1, y1 = x[0],y[0]
                    x2,y2 = x[1], y[1]
                    x3 ,y3= x[2], y[2]
                    x4 ,y4= x[3], y[3]
                    
                
          
                    cv2.circle(color_image, (x1, y1), 7, (255, 0, 255), 5)  
                    cv2.circle(color_image, (x2, y2), 7, (255, 0, 255), 5)
                    cv2.circle(color_image, (x3, y3), 7, (255, 0, 255), 5)
                    cv2.circle(color_image, (x4, y4), 7, (255, 0, 255), 5)
                    
                    detect_holes(x1,y1,w,h,color_image, 1)
                    detect_holes(x2,y2,w,h,color_image, 2)
                    detect_holes(x3,y3,w,h,color_image, 3)
                    detect_holes(x4,y4,w,h,color_image, 4)
                    #detect_holes(x2,y2,w,h,color_image)
                    
                    '''
                    top_left_x = int(x1 - w / 2) 
                    top_left_y = int(y1 - h / 2)
                    bottom_right_x = int(x1 + w / 2)
                    bottom_right_y = int(y1 + h / 2)
                    cv2.rectangle(color_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
                                      
                   # Ensure the coordinates are within the image boundaries
                    top_left_x = max(0, top_left_x)
                    top_left_y = max(0, top_left_y)
                    bottom_right_x = min(color_image.shape[1], top_left_x + w)
                    bottom_right_y = min(color_image.shape[0], top_left_y + h)

                    # Crop the image using slicing
                    cropped_image = color_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] 
                    cv2.imshow('Cropped Corner', cropped_image)
                    '''
                    
                    
                    

                  
                scale_factor = 0.5
                resized_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)

                # Display the images
   
                cv2.imshow('Color Image', resized_image)
                #cv2.imshow('masked Image',masked_color_image)
                out_color.write(color_image)
 
        
               
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except RuntimeError as e:
                print("Error:", e)
                continue


finally:
    # Stop streaming
    pipeline.stop()
    out_color.release()
 
    cv2.destroyAllWindows()
    