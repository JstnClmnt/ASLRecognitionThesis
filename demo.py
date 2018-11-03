import cv2
import argparse
import chainer
from entity import params
from pose_detector import PoseDetector, draw_person_pose
from face_detector import FaceDetector, draw_face_keypoints
from hand_detector import HandDetector, draw_hand_keypoints
import time
import numpy as np
chainer.using_config('enable_backprop', False)
start_time = time.time()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--img', help='image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)

    # read image
    img = cv2.imread('data/halfbodyrico.jpg')

    # inference
    print("Estimating pose...")
    person_pose_array, _ = pose_detector(img)
    res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

    # each person detected
    for person_pose in person_pose_array:
        unit_length = pose_detector.get_unit_length(person_pose)
        data=person_pose
        data=np.delete(data,9,0)
        data=np.delete(data,9,0)
        data=np.delete(data,10,0)
        data=np.delete(data,10,0)        
        print(data)
        # hands estimation
        print("Estimating hands keypoints...")
        hands = pose_detector.crop_hands(img, person_pose, unit_length)
        if hands["left"] is not None:
            hand_img = hands["left"]["img"]
            bbox = hands["left"]["bbox"]
            hand_keypoints = hand_detector(hand_img, hand_type="left")
            res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
            print("Right")
            print(hand_keypoints)
            cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        if hands["right"] is not None:
            hand_img = hands["right"]["img"]
            bbox = hands["right"]["bbox"]
            hand_keypoints = hand_detector(hand_img, hand_type="right")
            res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
            print("Left")
            print(hand_keypoints)
            cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

    print('Saving result into result.png...')
    cv2.imwrite('result.png', res_img)


print("--- %s seconds ---" % (time.time() - start_time))