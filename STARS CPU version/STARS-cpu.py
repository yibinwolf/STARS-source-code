'''
:@Author: Wei Xu, Wang Congxiao, Bailang Yu
:@Date: 2024-11-06
'''
# Import the libraries used
import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import STARS_Library as lib

# Set workspace & Parameter Settings (do not have "_" in the workspace)
Work_Space = r"PATH"
Output_name = "STARS_XX"
Ref_pixel_num = 100
std_per = 1.4
YearMEAN_GMM_num = 14
Temporals_GMM_num = 4

# Read the data for filling
Input_File_Collection,Input_date_list,Input_spectrum_Collection,Target_spectrum_data,\
Target_data,Target_date,REAL_data,cloud_data,cloud_indexs,Annual_data,Annual_GMM_data,\
Temporals_GMM_data,whole_rows,whole_cols,whole_Geotrans,proj,band_num,output_path \
= lib.STARS_Read_Data(Work_Space,YearMEAN_GMM_num,Temporals_GMM_num)

# Initializes the output array
Predict_output = Target_data.copy()

# Gets the background value in the image
Background_Value = np.min(REAL_data[REAL_data != 0])

# Cycle to fill each pixel to be filled
for cloud_index in tqdm(cloud_indexs,desc='Progress of pixels to be filled'):

    #--------------------#
    # Search for similar pixels
    similar_indexs,similar_pixel_flag,Quality_Flag = lib.Cal_Similar_Indexs(Annual_GMM_data,Temporals_GMM_data,Input_spectrum_Collection,Target_spectrum_data,cloud_index,cloud_data,Ref_pixel_num,Annual_data,std_per)
    if similar_pixel_flag == True:
        #--------------------#
        # Calculate the time series root-mean-square error of similar pixels
        TRMSD = lib.Cal_TRMSD(similar_indexs,Target_data,Input_File_Collection,cloud_index)
        # Calculate the spatial distance of similar pixels
        SD = lib.Cal_SD(similar_indexs,cloud_index)
        # Calculate the weight value of similar pixel
        Weight = lib.Cal_Weight(TRMSD, SD)
    
        #--------------------#
        # Calculate the standard deviation of DN values for each image
        SDRD = lib.Cal_SDRD(similar_indexs,Input_File_Collection,Target_data)
        # Calculate time distance for each image
        TD = lib.Cal_TD(Input_date_list,Target_date)
        # Calculate the reliability of each image
        Reliability = lib.Cal_Reliability(SDRD, TD)
    
        #--------------------#
        # Calculate the temporal predictor
        NTL_TP = lib.Cal_NTL_TP(Reliability,similar_indexs,Target_data,Weight,Input_File_Collection,cloud_index)
        # Calculate the spatial predictor
        NTL_SP = lib.Cal_NTL_SP(Target_data,similar_indexs,Weight)
        # Synthesis of spatiotemporal predicted values
        NTL_Predict = lib.Cal_NTL_Predict(NTL_SP,NTL_TP,TRMSD,SDRD)

        for ind in range(NTL_Predict.shape[0]):
            if NTL_Predict[ind] < Background_Value:
                NTL_Predict[ind] = Background_Value

        for i in range(NTL_Predict.shape[0]):
            Predict_output[i,cloud_index[0],cloud_index[1]] = NTL_Predict[i]
        
    else:
        
        for i in range(Target_data.shape[0]):
            Predict_output[i,cloud_index[0],cloud_index[1]] = np.mean(Input_File_Collection[:,i,cloud_index[0],cloud_index[1]])
# Output the filled tif image
lib.write_tif(os.path.join(output_path,Output_name+".tif"),Predict_output,whole_Geotrans,proj,gdal.GDT_Float32)