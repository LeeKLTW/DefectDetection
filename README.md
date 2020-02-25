# Steel Defect Detection Project

This is a project that use U-Net to predict and classify the defect regions of steel images with Kaggle dataset [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/overview/evaluation)  
  

## Explore data
The defect masks of the steel images are encoded using Run-length encoding. First we decoded the labels to masks and indicated the defect regions on the images.  
![1](https://github.com/RocioLiu/DefectDetection/blob/master/assets/0002cc93b.jpg)
![2](https://github.com/RocioLiu/DefectDetection/blob/master/assets/0007a71bf.jpg)
![3](https://github.com/RocioLiu/DefectDetection/blob/master/assets/000a4bcdd.jpg)
![4](https://github.com/RocioLiu/DefectDetection/blob/master/assets/000f6bf48.jpg)
![5](https://github.com/RocioLiu/DefectDetection/blob/master/assets/0014fce06.jpg)
  
  
## Training
We built a U-Net model
![6](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)   
  
  
