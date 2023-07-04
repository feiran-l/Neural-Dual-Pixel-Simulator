## Training & Test & Inferring using our Neural-DP-Simulator


### Train Neural-DP-Simulator from scratch
1. Download the dataset following the main README.md 
1. Run `train_dp_simulator.ipynb`, remember to change the data directory arg therein

### Use our pretrained DP-Simulator checkpoint
* Download the checkpoint from [here](https://drive.google.com/file/d/1tkDPaH2vN4G8Iq4btJ9loY42fYbuK3Fr/view?usp=sharing) and put it to `pretrained`
* Run `test_dp_simulator.ipynb` for testing on our DP5K dataset.
* Run `render_from_hypersim.ipynb` for rendering DP images from off-the-shelf RGBD dataset. `example_hypersim_data/` provides 3 example RGBD frames from the [Hypersim dataset](https://github.com/apple/ml-hypersim). 

