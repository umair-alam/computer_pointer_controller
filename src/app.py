from argparse import ArgumentParser
import logging
import cv2
import time
import os
from face_detection import facedetection
from facial_landmarks_detection import faciallandmarks
from gaze_estimation import gazeestimator
from head_pose_estimation import headpose
from input_feeder import InputFeeder
from mouse_controller import MouseController

#Model paths

model_1 = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
model_2 = "models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001"
model_3 = "models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009"
model_4 = "models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002"

def build_argparser():

    parser = ArgumentParser()
    parser.add_argument("-m_FD", "--face_detection", default=model_1, type=str, help = "Path to pre-trained Face Detection model")
    parser.add_argument("-m_HP", "--head_pose_estimation", default=model_2, type=str, help = "Path to pre-trained Head Pose Estimation Model")
    parser.add_argument("-m_LM", "--facial_landmarks_detection", default=model_3, type=str, help = "Path to pre-trained Facial Landmarks Detection Model")
    parser.add_argument("-m_GE", "--gaze_estimation", default=model_4 , type=str, help = "Path to pre-trained Gaze Estimation Model")
    parser.add_argument("-i", "--input", required=False, default="/bin/demo.mp4", type=str, help="Input File Path")
    parser.add_argument("-it", "--input_type", default="cam", help = "Input type: video or cam")
    parser.add_argument("-d", "--device", type=str, default="GPU", help="Device to run iinference on: Default CPU.")
    parser.add_argument("--extensions", default=None, help = "Any extensions for the selected device")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5, help="Thershold value for filtering: Default:0.5")
    return parser

class pointercontroller:

    def __init__(self, args):

        
        self.fd = facedetection(args.face_detection, args.device, args.extensions, args.prob_threshold)
        self.hp = headpose(args.head_pose_estimation, args.device, args.extensions)
        self.lm = faciallandmarks(args.facial_landmarks_detection, args.device, args.extensions)
        self.ge = gazeestimator(args.gaze_estimation, args.device, args.extensions)
        
        #loading the models and computing the loading time taken by each model
        
        loadtime_fd = time.time()
        self.fd.load_model()
        loadtime_fd = time.time()-loadtime_fd
        loadtime_hp = time.time()
        self.hp.load_model()
        loadtime_hp = time.time()-loadtime_hp
        loadtime_lm = time.time()
        self.lm.load_model()
        loadtime_lm = time.time()-loadtime_lm
        loadtime_ge = time.time()
        self.ge.load_model()
        loadtime_ge = time.time()-loadtime_ge
        
        #check for input type
        
        if(args.input_type == "cam"): self.input_feed = InputFeeder(args.input_type)
        else: self.input_feed = InputFeeder(args.input_type, args.input)
        self.input_feed.load_data()
        self.pointer_controller = MouseController('high', 'fast')
        
        print("Face Detection Model Load Time:", loadtime_fd)
        print("Head Pose Estimation Model Load Time:", loadtime_hp)
        print("Facial Landmarks Detection Model Load Time:", loadtime_lm)
        print("Gaze Estimation Model Load Time:", loadtime_ge)

    def run(self):
        total_inferencetime = []
        
        
        for batch in self.input_feed.next_batch():
            if batch is None:
                break
            inferencetime = time.time()
            cr_image, coords = self.fd.predict(batch)
            if cr_image is None:
                logger.error('NO FACE DETECTED')
                continue
            else:
                print("FACE DETECTED")
                left_eye_image, right_eye_image = self.lm.predict(cr_image)
                head_pose_angles = self.hp.predict(cr_image)
                vector = self.ge.predict(left_eye_image, right_eye_image, head_pose_angles)
                total_inferencetime.append(time.time() - inferencetime)
                batch = cv2.rectangle(batch, (coords[0],coords[1]), (coords[2],coords[3]), (0,255,255), thickness = 2)
                cv2.imshow("Output Window", batch)
                k = cv2.waitKey(60)
                self.pointer_controller.move(vector[0][0], vector[0][1])
                if k == 27: break
        
        self.input_feed.close()
        cv2.destroyAllWindows()
        print("Average of total inferences time:", sum(total_inferencetime) / len(total_inferencetime))

def main():
    args = build_argparser().parse_args()
    control = pointercontroller(args)
    control.run()
if __name__ == '__main__':
    main()