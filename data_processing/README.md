

### How we process the captured data

1. Use Canon's [Digital Photo Professional](https://cweb.canon.jp/eos/software/dpp4.html) software to extract the two DP-views from `.cr2` RAW. [This repo](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) provides an excellent tutorial on how to use it.  

2. Sequentially run `step-1...py` to `step-5...py` to get the final processed data 

3. For the structure-light-stereo system for depth acquisition, we have made it into an independent repo (though slightly different than here). Please click [here](https://github.com/SILI1994/Structured-light-stereo) for reference and environment setup. 
