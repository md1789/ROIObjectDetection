# ROIObjectDetection
# Steps to Reproduce Results From Scratch
If you do not have Anaconda currently installed on your PC, install it now. It is easiest to use Jupyter Notebook to train locally.

Download and extract [train_test_dataset_three_classes.zip](https://drive.google.com/file/d/1PdIOLI4-StPfIa64Rqu6EU8_9h2rgot6/view?usp=drive_link)

Go to [Roboflow](https://app.roboflow.com/) and create an account.

On the Roboflow Projects page, click Create New Project.

You will see this screen.
![image](https://github.com/md1789/ROIObjectDetection/assets/35352001/d1badbce-dc72-49c4-b201-482c36a54107)
Name your project. Fill in the annotation group box with these three classes: stent, bolus, catheter. The order does not matter. Now, click Create Public Project.

You should now see this screen.
![image](https://github.com/md1789/ROIObjectDetection/assets/35352001/dd9e897c-2eb2-4c55-969d-b12be6a32302)
Click Upload Data then click Select Folder and upload the folder you previously extracted.

Roboflow should automatically annotate the dataset based on the annotation group labels you provided.

Now you should be able to adjust the train-test-validation split percentages in Generate, as well as adjust pre-processing and augmentation steps. For the first trial, we want to see the performance on the raw annotated dataset, without any pre-processing or augmentations. Once you have created this dataset version. You will create another with these modifications:
![image](https://github.com/md1789/ROIObjectDetection/assets/35352001/819270cc-a2ea-4121-a303-14ea2b31b6e3)

To use this dataset on Yolov5 or Yolov7, click Export Dataset in the Versions page. Now select Yolov5 PyTorch from the following list:
![image](https://github.com/md1789/ROIObjectDetection/assets/35352001/28c29614-54fb-4573-a067-42c75893eaad)

You can use the Yolov7 PyTorch dataset version for Yolov7 if you prefer, but the Yolov5 PyTorch dataset version is the same type. In other words, it should not matter which one you use because they are the same file type. I did not test the Yolov7 PyTorch format, however, so there may be some subtle difference in the generated datasets.

To use this dataset on DeTr, you will need to export it to a COCO format.

Once you have created your datasets, you need to set up your Anaconda environment.

First, make sure cudatoolkit and cudnn are installed. If they are not, download here:

[Cudatoolkit](https://www.anaconda.com/products/distribution)

[CUDNN](https://developer.nvidia.com/cuda-downloads)

The cudatoolkit and cudnn versions you install don't particularly matter because cuda has backwards compatibility, but if you have an older version cudatoolkit or cudnn installed, look at the compatibility matrix here to be sure:

[Cuda Compatibility Matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html)

In Anaconda Terminal -> conda create --name [environment name]

conda activate [environment name]

Now, install the following dependencies:

conda install jupyter

pip install ipywidgets -- for yolov5 and yolov7

pip install ultralytics==8.0.9 (note: this is a version I found to be compatible with my torch, torchvision, and torchaudio versions below, but it is not recent) -- for yolov5 and yolov7

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 (note: you can use other versions of pytorch, but you must make sure that they are cuda-enabled and find the comaptibile torchvision, torchaudio, and ultralytics versions)

pip install matplotlib

pip install scikit_learn

pip install pandas

pip install -i https://test.pypi.org/simple/ supervision==0.3.0 --for DeTr

pip install pytorch_lightning==2.0 --for DeTr

pip install transformers --for DeTr

pip install timm --for DeTr

pip install pycocotools --for DeTr

pip install -q coco_eval --for DeTr

pip install datasets==2.15

pip install roboflow
