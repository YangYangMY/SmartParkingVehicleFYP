# SmartParkingVehicleFYP
 A final year project going under the topic of Smart Parking with Double Parking Tracking System.
 
## Hardware and Software Requirements
1. CPU: Minimum of Intel Core-i5
2. RAM: Minimum of 16GB RAM
3. Graphic Card:  MUST HAVE Nvidia Graphic Cards
4. Storage: Minimum of 20 GB free space
5. Camera: Minimum of 1080p resolution
## Software Requirements
1. OS: Window 10 and above
2. Pycharm
3.Conda environment (recommended)
4. Microsoft Excel

### CUDA Installation
1. Check if your computer has a CUDA-capable GPU. You can do this by going to the NVIDIA website and searching for your GPU model. If your GPU is listed, then it is CUDA-capable.
2. Download the NVIDIA CUDA Toolkit. You can download the CUDA Toolkit from the NVIDIA website. Select the appropriate version for your operating system and architecture.
3. Install the NVIDIA CUDA Toolkit. Once you have downloaded the CUDA Toolkit, double-click on the installer file to start the installation process. Follow the on-screen instructions to complete the installation.
4. Set the CUDA environment variables. Once the CUDA Toolkit is installed, you need to set the CUDA environment variables. This will tell your computer where to find the CUDA Toolkit files. To set the CUDA environment variables, follow these steps:
a. Open the Control Panel.
b. Click on "System and Security".
c. Click on "System".
d. Click on "Advanced system settings".
e.Under "Environment Variables", click on "New".
f. In the "Variable name" field, type "CUDA_PATH".
g. In the "Variable value" field, type the path to the CUDA Toolkit installation directory.
h. Click "OK".
i. Click on "New" again.
j. In the "Variable name" field, type "CUDA_BIN_PATH".
k. In the "Variable value" field, type the path to the CUDA Toolkit bin directory.
l. Click "OK".
m. Click "OK" to close the Environment Variables window.
6. Installing all necessary packages
7. Download and unzip the code.
8. Open the code in Pycharm.
9. Install all the necessary libraries in requirements.txt through the code:
   ‘pip install -r requirements.txt’
10. For PyTorch, you need to check the CUDA version you have installed and install the correct version of Pytorch through the following link:
   ‘https://pytorch.org/get-started/locally/’
11. fter everything is installed, you can run the following code to check if PyTorch and CUDA is installed correctly:
    print (torch.cuda.is_available()) 
12. If it returns True, this means you have installed successfully.

### Run code directly from Pycharm

Run the main file directly in Main.py to start all the process automatically.
Wait for a few seconds, Microsoft Excel with all the other 4 window frames will be  shown.
You can check the live updates of Microsoft Excel through the opened file.
This page is intentionally left blank to indicate the back cover. Ensure that the back cover is black in color.
