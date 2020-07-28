# Computer Pointer Controller

### Description
The computer pointer controller project is the part of Intel Edge AI for IoT Developers Nano Degree offered by Udacity. The purpose of this project is to control computer pointer with eye gaze and head pose angles. The pretrained models were selected which can run parallel to other models, all the selected models are from Open Model Zoo. The model pipeline depicts the the usage of output of one model into other models to perform predictions and controlling the mouse pointer by utilising pyautogui python module, here is the pipeline diagram for more insight. <br><br>
![Pipeline](https://github.com/umair-alam/computer_pointer_controller/blob/master/resources/pipeline.png)

## Project Set Up and Installation

### Installation of OpenVINO on Ubuntu 18.04 LTS

Update and upgrade Ubuntu
~~~
sudo apt-get update
sudo apt-get upgrade
~~~
Install required dependencies
~~~
$ sudo apt install build-essential cmake git pkg-config libgtk-3-dev \ libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \ libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \ gfortran openexr libatlas-base-dev python3-dev python-numpy \ libtbb2 libtbb-dev libdc1394–22-dev
~~~
Creating an opencv_build directory, cd into it
~~~
$ mkdir ~/opencv_build && cd ~/opencv_build
~~~
Clone the OpenCV and OpenCV contrib repositories:
~~~
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd ~/opencv_build/opencv
~~~
Create a build directory
~~~
$ mkdir build && cd build
~~~
Now setup the build with CMake:
~~~
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D CMAKE_INSTALL_PREFIX=/usr/local \ -D INSTALL_C_EXAMPLES=ON \ -D INSTALL_PYTHON_EXAMPLES=ON \ -D OPENCV_GENERATE_PKGCONFIG=ON \ -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \ -D BUILD_EXAMPLES=ON ..
$ make -j(number of cores your CPU has), e.g. make -j2 or make -j4
$ sudo make install
~~~
Now to install OpenVINO
Register and download OpenVINO from the official website
Change directories to your downloads directory 
~~~
$ cd ~/Downloads
~~~
Untar the file using the following command and move into the directory
~~~
$ tar -xvf l_openvino_toolkit_p_2020.1.023.tgz
$ cd l_openvino_toolkit_p_2020.1.023
~~~
Install using a GUI Installation
~~~
$ sudo ./install_GUI.sh

Openvino will be installed on the /opt/intel/l_openvino_toolkit_p_2020.1.023 directory
~~~
Configure the model optimizer and add dependencies.
~~~
$ cd /opt/intel/openvino/install_dependencies

$ sudo -E ./install_openvino_dependencies.sh

$ source /opt/intel/openvino/bin/setupvars.sh command.
configure the model optimizer by changing to the prerequisites directory with $ cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites to start the process. Then run the script to configure the model optimizer with $ sudo ./install_prerequisites.sh. If you want to install only a specified model use the $ sudo ./install_prequisites_<model name>.sh command instead.
~~~

#### Verify the installation

Go to the Inference Engine demo directory and run the image classification verification script
~~~
$ cd /opt/intel/openvino/deployment_tools/demo
$ ./demo_squeezenet_download_convert_run.sh.
~~~
### Project Dependencies
As stated in requirements.txt file, following are pre-requisites

image==1.5.27  
ipdb==0.12.3  
ipython==7.10.2  
numpy==1.17.4  
Pillow==6.2.1  
requests==2.22.0  
virtualenv==16.7.9  

#### Install and creat python virtual environment

~~~
$ pip3 install virtualenv
$ virtualenv venv
$ source venv/bin/activate

~~~
#### Activate OpenVINO enivromrent 
~~~
$ source /opt/intel/openvino/bin/setupvars.sh
~~~
#### Install project dependencies
~~~
$ pip3 install -r requirements.txt
~~~
### Download required models

#### face-detection-adas-binary-0001 

``` 
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 -o ~/project_pointer/models/

```
#### head-pose-estimation-adas-0001

``` 
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 -o ~/project_pointer/models/

```
#### landmarks-regression-retail-0009

``` 
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 -o ~/project_pointer/models/

```

#### gaze-estimation-adas-0002 

``` 
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 -o ~/project_pointer/models/

```
### Project Directory Structure

```bash
.
├── bin
│   └── demo.mp4
├── models
│   └── intel
│       ├── face-detection-adas-binary-0001
│       ├── gaze-estimation-adas-0002
│       ├── head-pose-estimation-adas-0001
│       └── landmarks-regression-retail-0009
├── resources
│   └── pipeline.png
├── src
│   ├── app.py
│   ├── face_detection.py
│   ├── facial_landmarks_detection.py
│   ├── gaze_estimation.py
│   ├── head_pose_estimation.py
│   ├── input_feeder.py
│   └── mouse_controller.py
├── README.md
└── requirements.txt
```

## Demo
Open terminal and initialize virtual environment and openvino environment by typing the following commands
```
$ source venv/bin/activate
$ source /opt/intel/openvino/bin/setupvars.sh
```
cd into the project directory and type
```
$ python3 src/app.py
```
The command line arguments are defined by default and the model paths are also defined, therefore no need to specify the command line arguments. The details of the arguments is given in the Documentation tab.

## Documentation
There are following command line arguments that is supported by this project

```
usage: app.py [-h] [-m_FD FACE_DETECTION] [-m_HP HEAD_POSE_ESTIMATION]
              [-m_LM FACIAL_LANDMARKS_DETECTION] [-m_GE GAZE_ESTIMATION]
              [-i INPUT] [-it INPUT_TYPE] [-d DEVICE]
              [--extensions EXTENSIONS] [-pt PROB_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -m_FD FACE_DETECTION, --face_detection FACE_DETECTION
                        Path to pre-trained Face Detection model
  -m_HP HEAD_POSE_ESTIMATION, --head_pose_estimation HEAD_POSE_ESTIMATION
                        Path to pre-trained Head Pose Estimation Model
  -m_LM FACIAL_LANDMARKS_DETECTION, --facial_landmarks_detection FACIAL_LANDMARKS_DETECTION
                        Path to pre-trained Facial Landmarks Detection Model
  -m_GE GAZE_ESTIMATION, --gaze_estimation GAZE_ESTIMATION
                        Path to pre-trained Gaze Estimation Model
  -i INPUT, --input INPUT
                        Input File Path
  -it INPUT_TYPE, --input_type INPUT_TYPE
                        Input type: video or cam
  -d DEVICE, --device DEVICE
                        Device to run iinference on: Default CPU.
  --extensions EXTENSIONS
                        Any extensions for the selected device
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Thershold value for filtering: Default:0.5
```

## Benchmarks

### Models Loading Time with Device CPU

**Hardware Specifications**  
*Intel® Core™ m3-7Y30 CPU @ 1.00GHz × 4*  
*Memory: 8 GB DDR4*  
****
|Models| FP16|FP32|INT8|
|----------------|---------------------|-------------------|-------------------|
| Face Detection | 0.19115376472473145  |0.2052755355834961 | 0.1891341209411621|
| Facial Landmarks Detection | 0.057404518127441406 |0.05697441101074219 | 0.2378244400024414| 
| Head Pose Estimation | 0.11118078231811523 |0.06965756416320801 | 0.7631175518035889|
| Gaze Estimation | 0.10722184181213379 |0.09084081649780273 | 0.5559959411621094|

*Total Average Inference Time (FP16): 0.03079759081204732 ms *  
*Total Average Inference Time (FP32): 0.03198266798450101 ms *  
*Total Average Inference Time (INT8): 0.027267708497888902 ms *  

### Models Loading Time with Device GPU  

**Hardware Specifications**  
*Intel® HD Graphics 615 (GT2)*   
*Memory: 8 GB DDR4*  

|Models| FP16|FP32|INT8|
|----------------|---------------------|-------------------|-------------------|
| Face Detection | 32.952716588974  |34.62701606750488 | 33.124125719070435|
| Facial Landmarks Detection | 4.6950366497039795 |5.318625211715698 | 7.4481658935546875| 
| Head Pose Estimation | 7.394354581832886 |7.312445402145386 | 8.990448713302612|
| Gaze Estimation | 8.307527780532837 |8.132406234741211 | 10.664573907852173|

*Total Average Inference Time (FP16): 0.04709758429691709 ms *  
*Total Average Inference Time (FP32): 0.043879295650281404 ms *  
*Total Average Inference Time (INT8): 0.045806225608376896 ms *  

## Results
As from the above results, it is clear that the IGPU performance is very much low incase of model loading and the highest time it took to load Face detection model, however the loading time is different in three executions for same precision level. If we consider as the model loading is a one time task then the average inference time is good. 
Incase of CPU the model loading time of INT8 models are highest and I also experience some performance lags on my device but the FP16 models run very smoothly and perform better.

### Async Inference
As I have mentioned earlier about the hardware I am using, Intel Core m3 CPU is a low powered device therefore Async Inference helps in achieving somehow good performance in case of CPU, the execution thread can wait until the result is available therefore reducing burden on edge device.

### Edge Cases
1. Face detection can be done from a particular distance (i.e. Face need to be closer to the camera to be able to detect)
2. Multiple face detection: The system detect multiple faces and the face closer to the camera is taken into consideration. 



### References:
1. Udacity Course work and project workspaces
2. Udacity knowledge hub
3. https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html
4. https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
5. https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
6. https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
7. https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

