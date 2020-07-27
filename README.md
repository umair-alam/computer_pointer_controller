# Computer Pointer Controller

### Description
The computer pointer controller project is a part of Intel Edge AI for IoT Developers Nano Degree offered by Udacity. The purpose of this project os to control computer pointer with eye gaze and head pose angles. The pretrained models were selected which can run parallel to other models, all the selected models are from Open Model Zoo. The model pipeline depicts the the usage of output of one model into other models to perform predictions, here is the pipeline diagram for more insight. <br><br>
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
<br><br><br>
### Project Dependencies
<br><br><br>
As stated in requirements.txt file, following are pre-requisites

image==1.5.27
ipdb==0.12.3
ipython==7.10.2
numpy==1.17.4
Pillow==6.2.1
requests==2.22.0
virtualenv==16.7.9

<br><br><br>
### Project Directory Structure
<br><br><br>
```bash
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


*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
