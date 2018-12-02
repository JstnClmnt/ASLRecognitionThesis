import numpy as np
import cv2
import argparse
import chainer
import time
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints
chainer.using_config('enable_backprop', False)
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASL Gesture Interpreter')
    parser.add_argument('--streamip', type=str, default=None, help='For IP camera compatibility - just enter your phone\'s IP; make sure there is no stream password set')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    args = parser.parse_args()

    cap = None
    if args.streamip == None:
        cap = cv2.VideoCapture(0)
        print("Source: INTERNAL WEBCAM")
    else:
        cap = cv2.VideoCapture("http://%s:8080/videofeed" % args.streamip)
        print("Source: IP CAMERA app @ %s" % args.streamip)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Camera resolution: %d x %d" % (width, height))
    print("--------------------------------------------------------------------")
    print("Real-Time 2D Sign Language to Text Interpreter Using Pose Estimation")
    print("with Part Affinity Fields and Hidden Markov Model")
    print("--------------------------------------------------------------------")
    print("USING PHASE")
    print("--------------------------------------------------------------------")
    print("Press 'q' to start recording.")

    out = None
    isRecording = False

    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret == True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if isRecording:
                    print("Recording has finished!")
                    break
                else:
                    isRecording = True
                    cv2.destroyAllWindows()
                    out = cv2.VideoWriter('outpy.mp4', fourcc, 30, (width, height))
                    print("Recording has started. Press 'q' to stop recording and start classifying.")

            if isRecording:
                # print("Writing...")
                out.write(frame)

            cv2.imshow("ASL Gesture Interpreter", frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture('outpy.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Amount of Frames:",amount_of_frames)
    cap.set(cv2.CAP_PROP_FPS, 5)
    ret, img = cap.read()
    counter=1
    df=pd.DataFrame(columns=["Head","Left","Right"])
    frame_tracker=int(amount_of_frames/12)
    framecounter=0
    print(frame_tracker)
    left=0
    right=0
    newdf=pd.read_csv("sample.csv")
    while ret:
        ret, img = cap.read()
        # get video frame
        if not ret:
            print("Failed to capture image")
            break
        person_pose_array, _ = pose_detector(img)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
        if (counter%frame_tracker==0):
            for person_pose in person_pose_array:
                unit_length = pose_detector.get_unit_length(person_pose)
                # hands estimation
                firstPerson=True
                if not firstPerson:
                    continue
                print("Estimating hands keypoints...")
                hands = pose_detector.crop_hands(img, person_pose, unit_length)
                if hands["left"] is not None:
                    hand_img = hands["left"]["img"]
                    bbox = hands["left"]["bbox"]
                    hand_keypoints = hand_detector(hand_img, hand_type="left")
                    for x in range(len(hand_keypoints)):
                        if(hand_keypoints[x]!=None):
                            hand_keypoints[x]=list(np.delete(hand_keypoints[x],2))
                            hand_keypoints[x]=[int(y) for y in hand_keypoints[x]]
                    res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                    left=hand_keypoints
                    cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
                else:
                    left=[[1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000]]

                if hands["right"] is not None:
                    hand_img = hands["right"]["img"]
                    bbox = hands["right"]["bbox"]
                    hand_keypoints = hand_detector(hand_img, hand_type="right")
                    for x in range(len(hand_keypoints)):
                        if(hand_keypoints[x]!=None):
                            hand_keypoints[x]=list(np.delete(hand_keypoints[x],2))
                            hand_keypoints[x]=[int(y) for y in hand_keypoints[x]]
                    res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                    right=hand_keypoints
                    cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
                else:
                    right=[[1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000], [1000,1000]]
                print("Body Pose")
                person_pose=np.delete(person_pose,9,0)
                person_pose=np.delete(person_pose,9,0)
                person_pose=np.delete(person_pose,10,0)
                person_pose=np.delete(person_pose,10,0)   
                person_pose=person_pose.tolist()
                for z in range(len(person_pose)):
                    if(person_pose[z]!=None):
                        person_pose[z]=list(np.delete(person_pose[z],2))
                        person_pose[z]=[int(a) for a in person_pose[z]]
                print(person_pose)
                print("Left")
                print(left)
                print("Right")
                print(right)
            cv2.imshow("result", res_img)
            head=person_pose
            for x in range(len(head)):
                if(head[x]==None):
                    head[x]=[1000,1000]
            pca = sklearnPCA(n_components=1)
            head=pca.fit_transform(head)
            dfhead=pd.DataFrame(data=head)
            dfhead=dfhead.T
            dfhead=dfhead.rename(columns={0:"head_1",1:"head_2",2:"head_3",3:"head_4",4:"head_5",5:"head_6",6:"head_7",7:"head_8",8:"head_9",9:"head_10",10:"head_11",11:"head_12",12:"head_13",13:"head_14"})
            for x in range(len(left)):
                if(left[x]==None):
                    left[x]=[1000,1000]
            pca = sklearnPCA(n_components=1)
            left=pca.fit_transform(left)
            dfleft=pd.DataFrame(data=left)
            dfleft=dfleft.T
            dfleft=dfleft.rename(columns={0:"left_1",1:"left_2",2:"left_3",3:"left_4",4:"left_5",5:"left_6",6:"left_7",7:"left_8",8:"left_9",9:"left_10",10:"left_11",11:"left_12",12:"left_13",13:"left_14",14:"left_15",15:"left_16",16:"left_17",17:"left_18",18:"left_19",19:"left_20",20:"left_21"})
            for x in range(len(right)):
                if(right[x]==None):
                    right[x]=[1000,1000]
            pca = sklearnPCA(n_components=1)
            right=pca.fit_transform(right)
            dfright=pd.DataFrame(data=right)
            dfright=dfright.T
            dfright=dfright.rename(columns={0:"right_1",1:"right_2",2:"right_3",3:"right_4",4:"right_5",5:"right_6",6:"right_7",7:"right_8",8:"right_9",9:"right_10",10:"right_11",11:"right_12",12:"right_13",13:"right_14",14:"right_15",15:"right_16",16:"right_17",17:"right_18",18:"right_19",19:"right_20",20:"right_21"})
            df2=pd.concat([dfhead, dfleft,dfright], axis=1)
            df2["frame"]=framecounter
            df2["gesture"]="Secret"
            df2["speaker"]="Rico"
            framecounter=framecounter+1
            df2["frame"]=df2["frame"].astype(int)
            newdf=newdf.append(df2,sort=False)
        else:
            cv2.imshow("result", img)
        counter=counter+1
        #print("Frame",counter)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break#print(df)
    cap.release()
    cv2.destroyAllWindows()
    newdf.to_csv("test.csv",index=False)