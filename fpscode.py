import cv2
import time

if __name__ == '__main__' :
 
    # Start default camera
    video = cv2.VideoCapture(0);
    video.set(cv2.CAP_PROP_FPS, 10)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per Second:",fps)
     
 
    # Number of frames to capture
    num_frames = 120;
     
     
    print("Capturing",num_frames,"frames")
 
    # Start time
    start = time.time()
     
    # Grab a few frames
    for i in range(0, num_frames) :
        ret, frame = video.read()
 
     
    # End time
    end = time.time()
 
    # Time elapsed
    seconds = end - start
    print("Time taken : ",seconds)
 
    # Calculate frames per second
    fps  = num_frames / seconds;
    print("Estimated frames per second : ",fps)
 
    # Release video
    video.release()