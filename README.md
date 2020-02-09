# Selective-regression-model
The codes of paper "Risk-Controlled Selective Prediction for Regression Deep Neural Network Models".
(We are still updating this code repository.)

### Tropic Cyclone (TC) Intensity Estimation
Dataset of Tropical Cyclone for Image-to-intensity Regression (TCIR) [^TCIR] was put forward by Boyo Chen, BuoFu Chen and Hsuan-Tien Lin. Please browse web page [TCIR](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/) for detail.
Single image in the TCIR dataset has the size of  201 (height) \* 201 (width) \* 4 (channels). Four channels are Infrared, Water vapor, Visible and Passive microwave, respectively. We just use Infrared and Passive microwave channels.

File TCIntensityEstimation has all the file about tropic cyclone intensity estimation problem, including source code, "how to get data" and a trianed model weights file.

- download source dataset (~13GB) and unzip

```
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ALL_2017.h5.tar.gz
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ATLN_EPAC_WPAC.h5.tar.gz
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-CPAC_IO_SH.h5.tar.gz
```
Unzip the compressed files and save the data into the dir Selective-regression-model/TCIntensityEstimation/data.

- preprocessing

```
cd Selective-regression-model/TCIntensityEstimation/Src
python preprocess.py 
```
This step need more than 32GB computer memory.

- running

```
python clean_rotated_Sel_PostNet.py
```

Trained model will be saved in dir Selective-regression-model/TCIntensityEstimation/result\_model/. In result\_model/, weightsV2-improvement-450.hdf5 is a trained model weight file.

### Apparent Age Estimation
File ApparentAgeEstimation has all the file about apparent age estimation problem, including  model .prototxt, mean file, "how to get data" and a trianed model weights file, validation dataset, test dataset and "how to download a trained model".

Apparent age estimation, which tries to estimate the age as perceived by other humans from a facial image, is different from the biological (real) age prediction. You could get more detail about dataset and model from [ApparentAgeEstimation](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) [^AGE]. The code of Apparent Age Estimation Model has a lot hard codes, which need to change depended on own enviroment.

- download validation and test dataset (face only)

```
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
```

- download trained caffe model

```
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
```

- running (need Caffe installed)

```
cd Selective-regression-model/ApparentAgeEstimation/
python LAPClassification.py
```



[^TCIR]: Boyo Chen, Buo-Fu Chen, and Hsuan-Tien Lin. Rotation-blended CNNs on a new open dataset for tropical cyclone image-to-intensity regression. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), August 2018.

[^AGE]: Rothe R, Timofte R, Van Gool L. Dex: Deep expectation of apparent age from a single image[C]//Proceedings of the IEEE international conference on computer vision workshops. 2015: 10-15.