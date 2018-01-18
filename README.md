## Learning Photography Aesthetics with Deep CNNs

#### Overview

This is an implementation of the paper [Learning Photography Aesthetics with Deep CNNs](https://arxiv.org/pdf/1707.03981.pdf)
in PyTorch.
By pooling the feature maps of the output of each ResNet block, we are able to use
gradCAM to visualize which parts of the image contribute to the aesthetics of the image.

The authors of the paper have a Keras implementation [here](https://github.com/gautamMalu/Aesthetic_attributes_maps) though the code is 
quite messy, and there were a few bugs I had to fix with their visualization code

#### Setup

To install the environment (assuming you have Anaconda installed) just do:
```conda env create -f environment.yml -n <env_name>```

Then activate the environment
```source activate <env_name>```

#### Training

The training images can be downloaded from [my Google Drive](https://drive.google.com/open?id=1YoffIa2sukWea5ITq4vPKTe_mv-Ra4df)

The data files are in `data/*.csv`

Training uses config file which is included, to train:

```python train.py --config_file_path config.json```

The training should take ~3 mins per epoch


Or if you don't want to train the model, you can go through to the notebook and do the visualization/evaluation there.
I have included a checkpoint from a pretrained model :)

#### Performance
```angular2html
ColorHarmony    0.503174
Content         0.597445
DoF             0.688919
Object          0.654278
VividColor      0.706658
score           0.716499
```

These results are better than the ones in the paper probably because I fine-tuned the weights of resnet50 in additional 
to training the FC layers connected to the GAP features from the resnet blocks. The reason was because I wasn't able to 
produce their results in the paper.

More performance measures can be seen in the notebook, for example the loss/epoch, the correlation/epoch etc.
#### Visualization 
The visualization code is in notebook/Pytorch Visualization-V2.ipynb

Here is an image picked randomly from the dataset:

[image1]: ./README_images/test_visualization.png

![alt_text][image1]

For fun I scraped some images off Instagram to see if my model was really making sense!

Here are the top 10 rated images using images scraped from NBA's Instagram account:

[image2]: ./README_images/nba_top_10.png

![alt_text][image2]

Here are the bottom 10 rated images:

[image3]: ./README_images/nba_bottom_10.png

![alt_text][image3]

Here are the top 10 rated images using images scraped from Dior's Instagram account:

[image4]: ./README_images/dior_top_10.png

![alt_text][image4]


Here are the bottom 10 rated images using images scraped from Dior's Instagram account:

[image5]: ./README_images/dior_top_10.png

![alt_text][image5]