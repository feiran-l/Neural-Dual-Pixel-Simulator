# Neural-DP-Simulator
[ICCP'23]: [Learning to Synthesize Photorealistic Dual-pixel Images from RGBD Frames](https://ieeexplore.ieee.org/document/10233839)

##### Authors: [Feiran Li*](https://sites.google.com/view/feiranlihomepage/home), [Heng Guo*](https://gh-home.github.io/), [Hiroaki Santo](https://sites.google.com/view/hiroaki-santo/), [Fumio Okura](http://cvl.ist.osaka-u.ac.jp/user/okura/), [Yasuyuki Matsushita](http://cvl.ist.osaka-u.ac.jp/en/member/matsushita/)   (*equal contribution)

## TODO:
1. We have only released the processed (i.e., RGB) data currently. The RAW images (~700GB) are too large and we are working on the release. 



## Dataset
![DP5K Dataset](teaser/dataset_summary.png)
* Download the processed RGB dataset: [16-bit version from DropBox](https://www.dropbox.com/sh/ym03faddftnkclw/AACXN_8hKrTl6mKQuDS9xRl3a?dl=0) or [8-bit version from Kaggle](https://www.kaggle.com/datasets/feiranli/dp5k-dataset), [16-bit version from OneDrive](https://bupteducn-my.sharepoint.com/personal/guoheng_bupt_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fguoheng%5Fbupt%5Fedu%5Fcn%2FDocuments%2FDataset%2FDP&ga=1). Note that the bit depth difference only applies to the RGB image and depth maps are all in 16 bit.  

* Each timestamp-named folder refers to one scenario. Within it:
    * `meta_data.h5` records various info such as focal length, focus distance, and calibration data
    * `cam0` and `cam1` contain the RGB DP images and depth maps of the left-right cameras, respectively. All RGB images are saved in 16-bit format. The depth maps are also saved in 16-bit format in mm unit. 

* `data_processing/` contains the code we used for processing the data acquired from the camera. 



## Our simulator
![Neural-DP-Simulator](teaser/our_render_from_rgbd.png)

`our_simulator/` contains the code for training & testing & inferring using our DP-Simulator 


<!-- ## Citation -->







