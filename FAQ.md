# Frequently asked questions

## Instalation problems

**While trying to use the library installed from source you encounter error: ``RuntimeError: CUDA error: no kernel image is available for execution on the device``**

This error indicates that you actually have pytorch version which does not have CUDA enabled. To solve that you should refer to [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). 


**While running/installing you encounter error: ``ImportError: libGL.so.1: cannot open shared object file: No such file or directory``**

You should download the following libraries (often already installed in basic distribution of Unix):
```
apt-get install ffmpeg libsm6 libxext6  -y
```

Source: [StackOverflow](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)
