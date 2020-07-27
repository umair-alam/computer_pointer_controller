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
Change directories to your downloads directory with $ cd ~/Downloads

l_openvino_toolkit_p_2020.1.023.tgz
Untar the file using the following command and move into the directory
~~~
$ tar -xvf l_openvino_toolkit_p_<version>.tgz
$ cd l_openvino_toolkit_p_2020.1.023
~~~
Install using a GUI Installation
~~~
$ sudo ./install_GUI.sh
~~~
Openvino will be installed on the /opt/intel/l_openvino_toolkit_p_2020.1.023 directory

10- The first part of the installation will be complete, next you have to set up the variables, configure the model optimizer and add dependencies.

11- Change to the install_dependencies directory with $ cd /opt/intel/openvino/install_dependencies

12- Run the script to download the needed dependencies with the following:

$ sudo -E ./install_openvino_dependencies.sh

13- Next set your variables with the

$ source /opt/intel/openvino/bin/setupvars.sh command.

Optionally you can add it to your .bashrc file in the last line. Use nano or vi to edit the file. Just add that line at the end and source it.

14- Next, configure the model optimizer by changing to the prerequisites directory with $ cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites to start the process. Then run the script to configure the model optimizer with $ sudo ./install_prerequisites.sh. If you want to install only a specified model use the $ sudo ./install_prequisites_<model name>.sh command instead.

15- To verify installation, go to the Inference Engine demo directory with $ cd /opt/intel/openvino/deployment_tools/demo.

16- Then run the image classification verification script with $ ./demo_squeezenet_download_convert_run.sh.

17- If it says execution successful then it has run correctly.
Image for post
Image for post

18- Next, run the Inference Pipeline verification script with $ ./demo_security_barrier_camera.sh. When it completes you should get an image that displays the resulting frame with detections rendered as bounding boxes and text. Your framerate will vary depending on the machine you have. Close the window to verify the installation.
Image for post
Image for post
Img src = https://docs.openvinotoolkit.org/latest/inference_pipeline_script_lnx.png

18- Congratulations, you have installed OpenVINO.
Image for post
Image for post


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
