# cuCBET
Cross Beam Energy Transfer using C++/CUDA

### Dependencies
- C++14
- CUDA 10.2+

### Usage
To compile use the command

> nvcc -arch={your_architecture} src/main.cu

then run 

> ./a.out

### Details
Currently the code does not fully compute the CBET. The code will output several files to a folder "outputs". To plot these files run

>python3 cuPlot.py 
