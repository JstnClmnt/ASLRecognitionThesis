import cv2
import argparse
import chainer
import time
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints
chainer.using_config('enable_backprop', False)
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)

    cap = cv2.VideoCapture('sign language/003_0.mpg')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Amount of Frames:",amount_of_frames)
    cap.set(cv2.CAP_PROP_FPS, 5)
    ret, img = cap.read()
    counter=0
    df=pd.DataFrame(columns=["Head","Left","Right"])
    left=0
    right=0
    while ret:
        ret, img = cap.read()
        # get video frame
        if not ret:
            print("Failed to capture image")
            break

        if counter%4==0:
            person_pose_array, _ = pose_detector(img)
            res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
            
            # each person detected
            for person_pose in person_pose_array:
                unit_length = pose_detector.get_unit_length(person_pose)
                #print(person_pose)
                # hands estimation
                hands = pose_detector.crop_hands(img, person_pose, unit_length)
                if hands["left"] is not None:
                    hand_img = hands["left"]["img"]
                    bbox = hands["left"]["bbox"]
                    hand_keypoints = hand_detector(hand_img, hand_type="left")
                    res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                    left=hand_keypoints
                    #print(hand_keypoints)
                    cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

                if hands["right"] is not None:
                    hand_img = hands["right"]["img"]
                    bbox = hands["right"]["bbox"]
                    hand_keypoints = hand_detector(hand_img, hand_type="right")
                    hand_keypoints=np.delete(hand_keypoints, 2, 2)
                    res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                    right=hand_keypoints
                    #print(hand_keypoints)
                    cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

                df2=pd.DataFrame({"Head":[person_pose],"Left":[left],"Right":[right]})
                df=df.append(df2)
                cv2.imshow("result", res_img)
                counter=0
        else:
            cv2.imshow("result", img)
            
        counter=counter+1
        print("Frame",counter)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(df)
    cap.release()
    cv2.destroyAllWindows()
