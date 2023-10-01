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
Check if your computer has a CUDA-capable GPU. You can do this by going to the NVIDIA website and searching for your GPU model. If your GPU is listed, then it is CUDA-capable.
Download the NVIDIA CUDA Toolkit. You can download the CUDA Toolkit from the NVIDIA website. Select the appropriate version for your operating system and architecture.
Install the NVIDIA CUDA Toolkit. Once you have downloaded the CUDA Toolkit, double-click on the installer file to start the installation process. Follow the on-screen instructions to complete the installation.
Set the CUDA environment variables. Once the CUDA Toolkit is installed, you need to set the CUDA environment variables. This will tell your computer where to find the CUDA Toolkit files. To set the CUDA environment variables, follow these steps:
Open the Control Panel.
Click on "System and Security".
Click on "System".
Click on "Advanced system settings".
Under "Environment Variables", click on "New".
In the "Variable name" field, type "CUDA_PATH".
In the "Variable value" field, type the path to the CUDA Toolkit installation directory.
Click "OK".
Click on "New" again.
In the "Variable name" field, type "CUDA_BIN_PATH".
In the "Variable value" field, type the path to the CUDA Toolkit bin directory.
Click "OK".
Click "OK" to close the Environment Variables window.
Installing all necessary packages
Download and unzip the code.
Open the code in Pycharm.
Install all the necessary libraries in requirements.txt through the code:
‘ pip install -r requirements.txt’
For PyTorch, you need to check the CUDA version you have installed and install the correct version of Pytorch through the following link:
‘https://pytorch.org/get-started/locally/’
After everything is installed, you can run the following code to check if PyTorch and CUDA is installed correctly:
print (torch.cuda.is_available()) 
If it returns True, this means you have installed successfully.

### Run code directly from Pycharm

Run the main file directly in Main.py to start all the process automatically.
Wait for a few seconds, Microsoft Excel with all the other 4 window frames will be  shown.
You can check the live updates of Microsoft Excel through the opened file.
This page is intentionally left blank to indicate the back cover. Ensure that the back cover is black in color.
