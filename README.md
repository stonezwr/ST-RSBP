#  Spike-Train level RSNNs Backpropagation (ST-RSBP) for SNNs
This repo is the CUDA implementation of SNNs trained the Spike-Train levelRSNNs Backpropagation, modified based on <a href="https://github.com/jinyyy666/mm-bp-snn">HM2-BP</a> for spiking neuron networks.

The paper.

Contact <stonezwr@gmail.com> if you have any questions or concerns.

# Dependencies and Libraries
* OpenCV
* CUDA (suggest CUDA 8.0)
* OpenMP

You can compile the code on windows or linux.   
##### SDK include path(-I)   
* linux: /usr/local/cuda/samples/common/inc/ (For include file "helper_cuda"); /usr/local/include/opencv/ (Depend on situation)        
* windows: X:/Program Files (x86) /NVIDIA Corporation/CUDA Samples/v6.5/common/inc (For include file "helper_cuda"); X:/Program Files/opencv/vs2010/install/include (Depend on situation)

##### Library search path(-L)   
>* linux: /usr/local/lib/   
>* windows: X:/Program Files/opencv/vs2010/install/x86/cv10/lib (Depend on situation)    
>
##### libraries(-l)      
>* opencv_core   
>* opencv_highgui   
>* opencv_imgproc   
>* opencv_imgcodecs (need for opencv3.0)  
>* openmp
>* ***cublas***   
>* ***curand***   
>* ***cudadevrt***  
>* ***cudacusparse***  
>* ***cudacurand*** 
>* ***cudacusolver*** 

# Installation

The repo requires [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 8.0+ to run.

Please install the opencv and cuda before hand.

Install CMake and OpenCV
```sh
$ sudo apt-get install cmake libopencv-dev 
```

Checkout and compile the code:
```sh
$ git clone https://github.com/jinyyy666/mm-bp-snn.git
$ cd mm-bp-snn
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```
##### GPU compute compatibility
* capability 6.0 for Titan XP, which is used for the authors. 
* Check the compatibility and change the CMAKE file before compile.


## Get Dataset
MNIST:
```sh
$ cd mnist/
$ ./get_mnist.sh
```
N-MNIST: Get the N-MNIST dataset by [N-MNIST](http://www.garrickorchard.com/datasets/n-mnist). Then unzip the ''Test.zip'' and ''Train.zip''. Run the matlab code: [NMNIST_Converter.m](https://github.com/stonezwr/ST-RSBP/tree/master/other_tools/nmnist)

N-Tidigits: [N-Tidigits](https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit), read the dataset using [N-Tidigits_Converter.py](https://github.com/stonezwr/ST-RSBP/tree/master/other_tools/NTidigits_Converter).

Fashion-MNIST: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

TI46: Download from [TI46](https://catalog.ldc.upenn.edu/LDC93S9) and preprocess it using Lyon ear model using [TI46_Lyon_ear.m](https://github.com/stonezwr/ST-RSBP/tree/master/other_tools/Lyon_ear_model)

## Run the code 
* MNIST 
```sh
$ ./build/CUDA-RSNN 1 1
```
* N-MNIST 
```sh
$ ./build/CUDA-RSNN 2 1
```
* Fashion Mnist
```sh
$ ./build/CUDA-RSNN 3 1
```
* TI46_Alpha
```sh
$ ./build/CUDA-RSNN 4 1
```
* TI46_Digits
```sh
$ ./build/CUDA-RSNN 5 1
```
* N-Tidigits
```sh
$ ./build/CUDA-RSNN 6 1
```

##### For Window user
Do the following to set up compilation environment.
* Install [Visual Stidio](https://www.visualstudio.com/downloads/) and [OpenCV](https://opencv.org/releases.html).
* When you create a new project using VS, You can find NVIDIA-CUDA project template, create a cuda-project.
* View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Device-> Code Generation-> compute_60,sm_60   
* View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Common-> Generate Relocatable Device Code-> Yes(-rdc=true) 
* View-> Property Pages-> Configuration Properties-> Linker-> Input-> Additional Dependencies-> libraries(-l)   
* View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Library search path(-L)  
* View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Include Directories(-I)  

# Notes
* The SNNs are implemented in terms of layers. User can config the SNNs by using configuration files in Config/
* The program will save the best test result and save the network weight in the file "Result/checkPoint.txt", If the program exit accidentally, you can continue the program form this checkpoint.
* The logs for the reported performance and the settings can be found in [Result](https://github.com/stonezwr/ST-RSBP/tree/master/Result) folder.
