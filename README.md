# Gradient-weighted Class Activation Mapping (Grad-CAM)
Deep neural networks have enabled exceptional breakthroughs in a variety of tasks. However, their lack of explanatory power makes them hard to interpret. Moreover, in the case of failure, they leave users wondering why! So, building models that are explainable is crucial to establish appropriate trust and confidence in users. Also, the explanatory power allows researchers to redirect their effort towards the main causing problem. 
Gradient-weighted Class Activation Mapping (Grad-CAM) is a class-discriminative localization technique for making any convolutional neural network model more transparent by producing visual explanations (Selvaraju et. al., 2017).

## Overview
Grad-CAM uses the gradient information flowing into the last convolutional layer of the model to obtain localization map and understand the importance of each pixel of the input image for a specific class. 
Let’s assume, ![](http://latex.codecogs.com/gif.latex?L_{Grad-CAM}^c\in\mathbb{R}^{u{\times}v})represents the localization map with width ![](http://latex.codecogs.com/gif.latex?u) and height ![](http://latex.codecogs.com/gif.latex?v) for class ![](http://latex.codecogs.com/gif.latex?c). To calculate ![](http://latex.codecogs.com/gif.latex?L_{Grad-CAM}^c), the gradient of the score for class ![](http://latex.codecogs.com/gif.latex?c) (before softmax), with respect to feature map ![](http://latex.codecogs.com/gif.latex?k), ![](http://latex.codecogs.com/gif.latex?A^{k}), of the last convolutional layer is calculated and global average pooled to obtain neuron importance weight, ![](http://latex.codecogs.com/gif.latex?a_k^c):
<p align="center">
<img src="http://latex.codecogs.com/gif.latex?a_k^c&space;=&space;\frac{1}{N}\sum_{i}\sum_{j}\frac{\partial&space;y^c}{\partial&space;A_{ij}^{k}}" title="a_k^c = \frac{1}{N}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A_{ij}^{k}}" />
</p>


Furthermore, a weighted combination of forward activation maps followed by ReLU is obtained:
<p align="center">
<img src="http://latex.codecogs.com/gif.latex?L_{Grad-CAM}^c&space;=ReLU\left&space;\left&space;(&space;\sum_{k}a_k^cA^k&space;\right&space;)" title="L_{Grad-CAM}^c =ReLU\left \left ( \sum_{k}a_k^cA^k \right )" />
</p>


This results in a coarse heatmap of the same size as the convolutional feature maps. Using ReLU allows us to capture features that have positive influence on the class of interest, i.e. pixels whose intensity should be increased in order to increase ![](http://latex.codecogs.com/gif.latex?y^c). Negative pixels are likely to belong to other categories in the image. Without ReLU, localization maps sometimes highlight more than just the desired class and achieve lower localization performance. 

## Results
Bellow images show Grad-CAM visualizations for two samples from CIFAR10. The first one belongs to category “deer” and the second one belongs to category “ship”:
<p align="center">
<img width="572" alt="" src="https://user-images.githubusercontent.com/20761298/65877354-d0f50f00-e3ce-11e9-9a7e-cc5f966c10db.png">
</p>
<p align="center">
<img width="572" alt="" src="https://user-images.githubusercontent.com/20761298/65877455-144f7d80-e3cf-11e9-9b21-035a566c4370.png">
</p>
From Grad-CAM visualizations, it can be concluded that the trained model looks for horn to identify a deer and exhaust pipe/mast to detect a ship. This provides explanation why the model failed to identify the right category in the following cases:
<p align="center">
<img width="572" alt="" src="https://user-images.githubusercontent.com/20761298/65877543-4e208400-e3cf-11e9-8f70-414d34ecd65c.png">
</p>
<p align="center">
<img width="572" alt="" src="https://user-images.githubusercontent.com/20761298/65877606-701a0680-e3cf-11e9-816d-a65318d962d5.png">
</p>

## Reference:

Salvaraju, R., Cogswell, M., Das, A., Vedantam, R., PArikh, D., and
Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *IEEE International Conference on Computer Vision (ICCV)*
(<https://arxiv.org/abs/1610.02391>)


