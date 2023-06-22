# depthai-superpoint
This repository is only for SuperPoint preview and performance evaluation on OAK cameras. Post-processing algorithms are not included here. The work here has been used in [RTAB-Map](https://introlab.github.io/rtabmap/) as part of a SLAM system.
```
mkdir build
cd build
cmake -D'depthai_DIR=depthai-core/build' ..
make
./depthai_superpoint ../blobs/superpoint_200x320_10shave.blob
```
![](screenshot.png)
