"""
Importing the required Modules
"""
import cv2
import numpy as np

"""
Get a VideoCapture object from video and store it in cap
""" 
cap = cv2.VideoCapture("Hallway 2.avi")


"""
Read first frame, then Scale and resize image
"""

ret, first_frame = cap.read()
 
resize_dim = 600
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)

"""
Convert to gray scale and Create mask
"""
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(first_frame)

"""
 Sets image saturation to maximum
"""
mask[..., 1] = 255


out = cv2.VideoWriter('video.mp4',-1,1,(600, 600))

while(cap.isOpened()):
    """
Frame-by-frame reading of the loaded video
"""
    ret, frame = cap.read()
    
    """
Convert new frame format`s to gray scale and resize gray frame obtained
"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

    """
    Calculate dense optical flow by Farneback method
     Ref: https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    """
Compute the magnitude and angle of the 2D vectors
"""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    """
Setting image hue according to the optical flow direction
"""
    mask[..., 0] = angle * 180 / np.pi / 2
    
    """
Setting image value according to the optical flow magnitude (normalized)
"""
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    """
Convert HSV to RGB (BGR) color representation
"""
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    """
Resize frame size to match dimensions
"""
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    """
Open a new window and displays the output frame
"""
    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
    cv2.imshow("Dense Model of optical flow", dense_flow)
    out.write(dense_flow)
    
    """
Update previous frame
"""
    prev_gray = gray
    
    """
Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
"""
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
"""
Releasing the Cap and destroying all windows
"""
cap.release()

cv2.destroyAllWindows()
