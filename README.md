# yolov4-darknet-autotag-easyone 
One Easy Process custom dataset YOLOv4 darknet model training and Autotagging of huge dataset with YOLOv4 custom model

# Easyone custom dataset YOLOv4 darknet model training

## Environment Setup
### Download darknet
```
git clone https://github.com/AlexeyAB/darknet.git
```

Make sure you have a GPU with CUDA support

### Makefile for GPU
```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
### Makefile for CPU
```
GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=1
AVX=0
OPENMP=1
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
### Compile darknet
```bash
make
```
### Download pre-trained weights
```
wget
```

## Training
### Create train.txt and test.txt
```
python create_train.py
```
### Create obj.names, obj.data and yolov4-custom.cfg
```
python create_config.py
```
### Start training
```
./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map
```
### Start training with multiple GPUs
```
./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map -gpus 0,1,2,3
```
### Start training with multiple GPUs and mixed precision
```
./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map -gpus 0,1,2,3 -mixed
```
### Start training with multiple GPUs and mixed precision and Tensor Cores
```
./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map -gpus 0,1,2,3 -mixed -cudnn_half
```

## Easyone Training

1. Place labeled dataset in `[custom name dataset]/dataset` folder
2. Run `one_process.py` and follow the instructions
    ```
    python one_process.py
    ```
# EasyOne Process will
- Create `train.txt` and `test.txt` files
- Create `obj.names`, `obj.data` files
- Create `anchors.sh` script
- Create `train.sh` script
#

3. For anchor box calculation, run `anchors.sh` in root folder
    ```
    sh anchors.sh
    ```
4. Copy the output and paste it in `yolov4-custom.cfg` file 3 times
5. For training, run `train.sh` in root folder
    ```
    sh train.sh
    ```

6. After training, you can find the trained model in `[custom name dataset]/backup` folder

# Autotag images with custom trained YOLOv4 darknet model

- Place images to be autotagged in the `autotag_dataset` folder
- In `autotag_model` folder :
    1. - Place the `yolov4-custom_best.weights` file
    2. - Place the `yolov4-custom.cfg` file
    3. - Place the `obj.data` file

## Run Auto-tagging Script
Run `python autotag.py` to autotag images
```bash
python autotag.py
```





