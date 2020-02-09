# Selective-regression-model
The code of paper Risk-Controlled Selective Prediction for Regressiondeep neural nwtwork models.
We are still preparing for it.

### Tropic Cyclone (TC) Intensity Estimation
Dataset of Tropical Cyclone for Image-to-intensity Regression (TCIR) [^TCIR] was put forward by Boyo Chen, BuoFu Chen and Hsuan-Tien Lin. Please browse web page [TCIR](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/) for detail.

Single image in the TCIR dataset has the size of  201 (height) \* 201 (width) \* 4 (channels). Four channels are Infrared, Water vapor, Visible and Passive microwave, respectively.

File TCIntensityEstimation has all the file about tropic cyclone intensity estimation problem, including source code, "how to get data" and a trianed model weights file.

- download source dataset and unzip

```
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ATLN_EPAC_WPAC.h5.tar.gz
```
- preprocessing

```
cd Selective-regression-model/Src
python preprocess.py
```
- running

```
```

### Apparent Age Estimation
File ApparentAgeEstimation has all the file about apparent age estimation problem, including  model .prototxt, mean file, "how to get data" and a trianed model weights file, validation dataset, test dataset and "how to download a trained model".


[^TCIR]: Boyo Chen, Buo-Fu Chen, and Hsuan-Tien Lin. Rotation-blended CNNs on a new open dataset for tropical cyclone image-to-intensity regression. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), August 2018.