import cv2
from openvino.inference_engine import IENetwork, IECore
import numpy as np

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class gazeestimator:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        #self.threshold = thershold
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.infer_request_handle = None
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        #raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
    
        #raise NotImplementedError

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye = self.preprocess_input(left_eye_image)
        right_eye = self.preprocess_input(right_eye_image)

        input_dict = {'left_eye_image':left_eye, 'right_eye_image':right_eye, 'head_pose_angles':head_pose_angles}
        infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1)==0: 
            return self.preprocess_output(infer_request_handle.outputs)
        #raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image = cv2.resize(image, (60,60))
        image = image.transpose((2,0,1))
        image = image.reshape((1, *image.shape))
        return image
        #raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gazevector = outputs[self.output_name]
        return gazevector
        #raise NotImplementedError
