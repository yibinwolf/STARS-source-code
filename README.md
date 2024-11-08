# STARS source code instructions 

## Background 

To address the cloud contamination present in Sustainable Development Goals Satellite 1 (SDGSAT-1) Glimmer Imagery (GLI) nighttime light (NTL) data , we develop a novel method named STARS (SpatioTemporal And spectRal gap filling method for SDGSAT-1 glimmer imagery) to fill gaps in SDGSAT-1 GLI NTL images and extend the available time series. By selecting similar pixels in terms of temporality, space, and spectrum, a series of similar pixels can be identified for filling cloud gaps. The filling is then conducted based on the weights of these similar pixels.

## Copyright

The principle and code of STARS have been applied for patent in China. The patent name is 《一种基于时空谱信息协同的遥感影像重建方法》and the patent code is 202410787191.4

The paper related to STARS has been submitted to Remote Sensing of Environment. Please refer to our paper when using it.

The code used in STARS is now open source on Github, limited to personal study and scientific research, and cannot be used for commercial purposes. Please comply with relevant laws and regulations.


## Details of application

### 1.Preprocessing

Before using STARS method, you need to identify the outline of the cloud with the code we provided (source code is located at Matlab Code -> Cloud identification which should be run in Matlab). And then the original SDGSAT-1 GLI NTL image requires a series of preprocessing (source code is located in the Preprocess.ipynb file in the Preprocess folder and mainly uses the Arcpy and Arosics library for preprocessing), including: (1) noise removal; (2) image mosaicing; (3) Reprojection; (4) Image registration；(5) maximum common subregion extraction; (6) Data snap and clip; (7) Cloud masking; (8) mean synthesis; (9) Calculation of SONDI index; (10) Copy related files to the specified path

### 2.STARS main section

The STARS algorithm is divided into two versions, the CPU version and the GPU version. The CPU version is located in the STARS CPU version folder, and the GPU version is located in the STARS GPU version folder. The relevant functions used in the STARS algorithm are located in the STARS_Library.py file, and you need to manually set the following parameters before running the code to perform filling.

```
Work_Space        =   "User-defined workspace"
Output_name       =   "User-defined output name"    
Ref_pixel_num     =   The number of similar pixels, int
std_per           =   Range of standard deviations, float
YearMEAN_GMM_num  =   The number of GMM clusters for the average composite image
Temporals_GMM_num =   The number of GMM clusters for time series luminance synthesis image
```

## Final 

All developers are welcome to participate in the project, you can go to Github to raise an Issue or PR. If you have any questions or suggestions, please contact us through email.