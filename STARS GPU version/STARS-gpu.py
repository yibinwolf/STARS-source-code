'''
:@Author: Wei Xu, Wang Congxiao, Bailang Yu
:@Date: 2024-11-06
'''
# Import the libraries used
import os
import cupy as cp
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

# Convert data to cupy format
Input_File_Collection_cupy = cp.asarray(Input_File_Collection)
Target_data_cupy = cp.asarray(Target_data)
cloud_data_cupy = cp.asarray(cloud_data)
Annual_GMM_data_cupy = cp.asarray(Annual_GMM_data)
Temporals_GMM_data_cupy = cp.asarray(Temporals_GMM_data)
Input_spectrum_Collection_cupy = cp.asarray(Input_spectrum_Collection)
Target_spectrum_data_cupy = cp.asarray(Target_spectrum_data)
Annual_data_cupy = cp.asarray(Annual_data)

# Cycle to fill each pixel to be filled
for cloud_index in tqdm(cloud_indexs,desc='Progress of pixels to be filled'):

    cloud_index_cupy = cp.asarray(cloud_index)

    #--------------------#
    # Search for similar pixels
    similar_indexs_cupy,similar_pixel_flag,Quality_Flag = lib.Cal_Similar_Indexs(Annual_GMM_data_cupy,Temporals_GMM_data_cupy,Input_spectrum_Collection_cupy,Target_spectrum_data_cupy,cloud_index_cupy,cloud_data_cupy,Ref_pixel_num,Annual_data_cupy,std_per)
    if similar_pixel_flag == True:
        #--------------------#
        # Calculate the time series root-mean-square error of similar pixels
        TRMSD_cupy = lib.Cal_TRMSD(similar_indexs_cupy,Target_data_cupy,Input_File_Collection_cupy,cloud_index_cupy)
        # Calculate the spatial distance of similar pixels
        SD_cupy = lib.Cal_SD(similar_indexs_cupy,cloud_index_cupy)
        # Calculate the weight value of similar pixel
        Weight_cupy = lib.Cal_Weight(TRMSD_cupy, SD_cupy)
    
        #--------------------#
        # Calculate the standard deviation of DN values for each image
        SDRD_cupy = lib.Cal_SDRD(similar_indexs_cupy,Input_File_Collection_cupy,Target_data_cupy)
        # Calculate time distance for each image
        TD = lib.Cal_TD(Input_date_list,Target_date)
        TD_cupy = cp.asarray(TD)
        # Calculate the reliability of each image
        Reliability_cupy = lib.Cal_Reliability(SDRD_cupy, TD_cupy)
    
        #--------------------#
        # Calculate the temporal predictor
        NTL_TP_cupy = lib.Cal_NTL_TP(Reliability_cupy,similar_indexs_cupy,Target_data_cupy,Weight_cupy,Input_File_Collection_cupy,cloud_index_cupy)
        # Calculate the spatial predictor
        NTL_SP_cupy = lib.Cal_NTL_SP(Target_data_cupy,similar_indexs_cupy,Weight_cupy)
        # Synthesis of spatiotemporal predicted values
        NTL_Predict_cupy = lib.Cal_NTL_Predict(NTL_SP_cupy,NTL_TP_cupy,TRMSD_cupy,SDRD_cupy)
        NTL_Predict = cp.asnumpy(NTL_Predict_cupy)

        for ind in range(NTL_Predict.shape[0]):
            if NTL_Predict[ind] < Background_Value or np.isnan(NTL_Predict[ind]):
                NTL_Predict[ind] = Background_Value

        Predict_output[:,cloud_index[0],cloud_index[1]] = NTL_Predict
        
    else:
        
        for i in range(Target_data.shape[0]):
            Predict_output[i,cloud_index[0],cloud_index[1]] = np.mean(Input_File_Collection[:,i,cloud_index[0],cloud_index[1]])
# Output the filled tif image
lib.write_tif(os.path.join(output_path,Output_name+".tif"),Predict_output,whole_Geotrans,proj,gdal.GDT_Float32)