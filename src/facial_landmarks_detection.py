import cv2
from openvino.inference_engine import IENetwork, IECore
import numpy as np
from math import sqrt
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class faciallandmarks:
    '''
    Class for the Facial Landmarks Detection Model.
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

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is for running predictions on the input image.
        '''
        self.processed_img = self.preprocess_input(image)
        input_dict = {self.input_name:self.processed_img}
        infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1)==0: return self.preprocess_output(infer_request_handle.outputs)
        #raise NotImplementedError
    def check_model(self):  
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference
        you might have to preprocess it. This function is where you can do that.
        '''
        self.image = image
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape((1, *image.shape))
        return image
        #raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = outputs[self.output_name]
        eye_square = sqrt((self.image.shape[0]*self.image.shape[1])/100)
        x_0, y_0, x_1, y_1 = output[0][0][0][0], output[0][1][0][0], output[0][2][0][0], output[0][3][0][0]
        x_0, y_0, x_1, y_1 = x_0*self.image.shape[1], y_0*self.image.shape[0], x_1*self.image.shape[1], y_1*self.image.shape[0]
        #x_2, y_2, x_3, y_3, x_4, y_4 = output[0][4][0][0], output[0][5][0][0], output[0][6][0][0], output[0][7][0][0], output[0][8][0][0], output[0][9][0][0]
        #x_2, y_2, x_3, y_3, x_4, y_4 = x_2*self.image.shape[1], y_2*self.image.shape[0], x_3*self.image.shape[1], y_3*self.image.shape[0], x_4*self.image.shape[1], y_4*self.image.shape[0]
        x_0, y_0, x_1, y_1, eye_square = int(x_0), int(y_0), int(x_1), int(y_1), int(eye_square)
        left_eye = self.image[y_0-eye_square:y_0+eye_square, x_0-eye_square:x_0+eye_square]
        right_eye = self.image[y_1-eye_square:y_1+eye_square, x_1-eye_square:x_1+eye_square]
        #all_coords = [x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]        
        return left_eye, right_eye, #all_coords
        #raise NotImplementedError
