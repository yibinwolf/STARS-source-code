'''
:@Author: Wei Xu, Wang Congxiao, Bailang Yu
:@Date: 2024-11-06
'''
import os
import glob
import time
import numpy as np
from numba import jit,prange
from osgeo import gdal
from sklearn.mixture import GaussianMixture

#---------------------------------------------------------------------------------------#
# ----------------------------------Main function---------------------------------------#
#---------------------------------------------------------------------------------------#

def STARS_Read_Data(Work_Space, YearMEAN_GMM_num, Temporals_GMM_num):
    #--------------------#
    # Path for NTL files used for filling, with R, G, B resolution of 40m; areas without values should be set to 0
    Input_path = os.path.join(Work_Space, "Input")

    # Path for NTL spectral feature files used for filling
    Input_spectrum_path = os.path.join(Work_Space, "Input_Spectrum")

    #--------------------#
    # Path for cloud mask file, with cloud regions set to 1 and non-value areas set to 0
    cloud_path = os.path.join(Work_Space, "Cloud", "Cloud.tif")

    # Path for annual composite data, with R, G, B resolution of 40m; areas without values should be set to 0
    Annual_path = os.path.join(Work_Space, "Annual")

    #--------------------#
    # Path for NTL files that need filling
    Target_path = os.path.join(Work_Space, "Target")

    # Path for spectral feature files of NTL data that need filling
    Target_spectrum_path = os.path.join(Work_Space, "Target_Spectrum")

    # Path for real NTL files (non-existent for real cloud filling)
    REAL_path = os.path.join(Work_Space, "REAL")

    #--------------------#
    # Create output folder if it does not exist
    if not os.path.exists(Work_Space + '\\' + "Output"):
        os.makedirs(Work_Space + '\\' + "Output")
    
    # Set output path
    output_path = os.path.join(Work_Space, "Output")

    #--------------------#
    # Read basic map information for output
    whole_rows, whole_cols, whole_Geotrans, proj, band_num = Get_Coordinate(os.path.join(Annual_path, "YearMEAN.tif"))

    # Read cloud image and cloud indices
    cloud_data, cloud_indexs = Read_Cloud(cloud_path)

    # Read annual composite data and GMM classification results for annual data
    Annual_data, Annual_GMM_data = Read_YearMean(Annual_path, whole_Geotrans, proj, YearMEAN_GMM_num)

    #--------------------#
    # Read NTL files that need filling
    Target_data_name = glob.glob(Target_path + "\\*.tif")
    Target_data = read_tif(Target_data_name[0])

    # Extract date of target image that needs filling
    Target_date = Target_data_name[0].split('_')[2]

    # Read real NTL files
    REAL_data_name = glob.glob(REAL_path + "\\*.tif")
    REAL_data = read_tif(REAL_data_name[0])

    # Read spectral features of NTL data that need filling
    Target_spectrum_name = glob.glob(Target_spectrum_path + "\\*.tif")
    Target_spectrum_data = read_tif(Target_spectrum_name[0])

    # Read collection of input files
    Input_File_Collection, Input_date_list = Get_Input_Files(Input_path, Target_data)
    Temporals_GMM_data = Temporals_GMM(Input_File_Collection, Temporals_GMM_num)
    write_tif(os.path.join(output_path, "RGB_GMM_n.tif"), Temporals_GMM_data, whole_Geotrans, proj, gdal.GDT_Int16)

    # Read spectral features of input file collection
    Input_spectrum_Collection = Get_Input_spectrum_Files(Input_spectrum_path, Target_data)

    return Input_File_Collection, Input_date_list, Input_spectrum_Collection, Target_spectrum_data, Target_data, Target_date, REAL_data, cloud_data, cloud_indexs, Annual_data, Annual_GMM_data, Temporals_GMM_data, whole_rows, whole_cols, whole_Geotrans, proj, band_num, output_path

#---------------------------------------------------------------------------------------#
# ---------------------------------Basic function---------------------------------------#
#---------------------------------------------------------------------------------------#
def read_tif(path):
    """
    Read the raster image
    """
    dataset = gdal.Open(path)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, cols, rows)
    del dataset 
    return im_data

def write_tif(newpath,im_data,im_Geotrans,im_proj,datatype):
    """
    Output the raster image (note: 0 is null!!)
    """
    if len(im_data.shape)==3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)
    
    if im_bands == 1: 
        new_dataset.GetRasterBand(1).WriteArray(im_data)
        new_dataset.GetRasterBand(1).SetNoDataValue(0)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i+1).WriteArray(im_data[i])
            new_dataset.GetRasterBand(i+1).SetNoDataValue(0)

    del new_dataset

#---------------------------------------------------------------------------------------#
# ------------------------------------Read file-----------------------------------------#
#---------------------------------------------------------------------------------------#

def Read_Cloud(cloud_path):
    """
    Read cloud image and cloud indexes
    """
    cloud_data = read_tif(cloud_path)
    cloud_indexs = np.argwhere(cloud_data == 1)
    return cloud_data, cloud_indexs

def Read_YearMean(Annual_path, whole_Geotrans, proj, YearMEAN_GMM_num):
    """
    Read the mean composite data and perform GMM classification
    """
    Annual_data = read_tif(os.path.join(Annual_path,"YearMEAN.tif"))
    Mask = np.where(Annual_data == 0, 0, 1)
    Mask_Annual_data = np.where(Mask == 1,Annual_data,-9999)

    if os.path.exists(os.path.join(Annual_path,"YearMEAN_GMM_"+str(YearMEAN_GMM_num)+".tif")):
        Annual_GMM_data = read_tif(os.path.join(Annual_path,"YearMEAN_GMM_"+str(YearMEAN_GMM_num)+".tif"))
    else:
        Annual_GMM_data = Cal_GMM(Mask_Annual_data,Annual_data.shape[0],YearMEAN_GMM_num+1)
        write_tif(os.path.join(Annual_path,'YearMEAN_GMM_'+str(YearMEAN_GMM_num)+'.tif'),Annual_GMM_data,whole_Geotrans,proj,gdal.GDT_Int16)

    return Annual_data, Annual_GMM_data

def Temporals_GMM(Input_File_Collection,n=4):
    """
    Get the evolution of the input files using GMM classification
    """
    Mask = np.where(Input_File_Collection == 0, 0, 1)
    Mask_Temporals_data = np.where(Mask == 1,Input_File_Collection,-9999)
    n_files, n_bands, n_rows, n_cols = Input_File_Collection.shape
    imgxybn = np.zeros(shape=(n_rows, n_cols, n_files, n_bands))
    for i in range(n_files):
        for band in range(n_bands):
            imgxybn[:,:,i,band] = Mask_Temporals_data[i,band,:,:]

    reshaped_imgxybn = imgxybn.reshape(n_rows * n_cols, n_bands * n_files)

    gmm = GaussianMixture(n_components=n+1, covariance_type='full')
    gmm.fit(reshaped_imgxybn)
    labels = gmm.predict(reshaped_imgxybn)

    labels_image = labels.reshape(n_rows,n_cols)
    labels_image = labels_image + 1

    return labels_image

def Cal_GMM(array,band_num,n=10):
    """
    Gaussian mixture model classification
    """
    imgxyb = np.zeros(shape=(array.shape[1], array.shape[2], array.shape[0]))
    for band in range(imgxyb.shape[2]):
        imgxyb[:,:,band] = array[band,:,:]
    img1d=imgxyb[:,:,:band_num].reshape((imgxyb.shape[0]*imgxyb.shape[1],imgxyb.shape[2]))

    gmm = GaussianMixture(n_components=n, covariance_type='full')
    gmm.fit(img1d)
    labels = gmm.predict(img1d)
    labels_image = labels.reshape(imgxyb[:,:,0].shape)
    labels_image = labels_image + 1

    return labels_image

def Get_Coordinate(Annual_path):
    """
    Read basic information
    """
    dataset = gdal.Open(Annual_path)
    band_num = dataset.RasterCount
    proj = dataset.GetProjection()
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    Geotrans = (dataset.GetGeoTransform())

    return rows,cols,Geotrans,proj,band_num

def Get_Input_Files(Input_path,Target_data):
    """
    Get a four-dimensional array of input files
    """
    NTL_files_0 = os.listdir(Input_path)
    NTL_files = []
    for NTL_file in NTL_files_0:
        if NTL_file[-3:] == 'tif':
            NTL_files.append(NTL_file)
    file_num = len(NTL_files)
    File_Collection = np.zeros(shape=(file_num,Target_data.shape[0],Target_data.shape[1],Target_data.shape[2]))
    date_list = []
    for i in range(file_num):
        temp_array = read_tif(os.path.join(Input_path,NTL_files[i]))
        date_list.append(NTL_files[i][9:17])
        File_Collection[i,:,:,:] = temp_array

    return File_Collection,date_list

def Get_Input_spectrum_Files(Input_spectrum_path,Target_data):
    """
    Get the spectral signature of the input file
    """
    NTL_files_0 = os.listdir(Input_spectrum_path)
    NTL_files = []
    for NTL_file in NTL_files_0:
        if NTL_file[-3:] == 'tif':
            NTL_files.append(NTL_file)

    File_Collection = np.zeros(shape=(len(NTL_files),Target_data.shape[1],Target_data.shape[2]))
    for i in range(len(NTL_files)):
        temp_array = read_tif(os.path.join(Input_spectrum_path,NTL_files[i]))
        File_Collection[i,:,:] = temp_array

    return File_Collection

#---------------------------------------------------------------------------------------#
# ---------------------------------Filling function--------------------------------------#
#---------------------------------------------------------------------------------------#

def Cal_Similar_Indexs(Annual_GMM_data, Temporals_GMM_data, Input_spectrum_Collection, Target_spectrum_data, cloud_index, cloud_data, Ref_pixel_num, Annual_data, std_per=1):
    """
    Obtain similar pixel indices.
    The flag `similar_pixel_flag` indicates the number of similar pixels: 
    True if the number of similar pixels is greater than 0, False if it equals 0.
    If `similar_pixel_flag` is False, the filling step is skipped, and an average value is used for filling instead.
    """
    # Default to finding similar pixels
    similar_pixel_flag = True

    # Default quality flag is 0
    Quality_Flag = 0

    # Search for pixels in the same GMM classification
    same_GMM_type_indexs = _Search_Same_Type(cloud_index, Annual_GMM_data, Temporals_GMM_data, cloud_data)

    if (len(same_GMM_type_indexs) > 0):

        ### Same-type pixels found; proceed to calculate SONDI range
        SONDI_similar_indexs = _Search_SONDI_similar(Input_spectrum_Collection, cloud_index, same_GMM_type_indexs, Target_spectrum_data, std_per)

        if (len(SONDI_similar_indexs) > Ref_pixel_num):

            ### SONDI similar pixels exceed ref, continue with RMSD calculation
            Final_indexs = _Search_Similar(SONDI_similar_indexs, Annual_data, cloud_index, Ref_pixel_num)
            
            # Set quality flag to 1
            Quality_Flag = 1

        elif(0 < len(SONDI_similar_indexs) <= Ref_pixel_num): 

            ### SONDI similar pixels are fewer than ref; use all of them
            Final_indexs = SONDI_similar_indexs

            # Set quality flag to 2
            Quality_Flag = 2

        elif(len(SONDI_similar_indexs) == 0):

            ### SONDI filtering results in no pixels; recalculate RMSD for same-type pixels

            if len(same_GMM_type_indexs) > Ref_pixel_num:
                
                ### More same-type pixels than ref, select 30 from them
                Final_indexs = _Search_Similar(same_GMM_type_indexs, Annual_data, cloud_index, Ref_pixel_num)
                
                # Set quality flag to 3
                Quality_Flag = 3

            elif len(same_GMM_type_indexs) < Ref_pixel_num:
                
                ### Fewer same-type pixels than ref; use all of them
                Final_indexs = same_GMM_type_indexs
                                
                # Set quality flag to 4
                Quality_Flag = 4
            else:
                ### No similar pixels found; set to empty
                Final_indexs = []

                # Set `similar_pixel_flag` to False
                similar_pixel_flag = False

                # Set quality flag to 5
                Quality_Flag = 5
    else:

        ### No similar pixels found; set to empty
        Final_indexs = []

        # Set `similar_pixel_flag` to False
        similar_pixel_flag = False

        # Set quality flag to 5
        Quality_Flag = 5

    return Final_indexs, similar_pixel_flag, Quality_Flag


@jit(nopython=True)
def Cal_SD(similar_indexs,cloud_index):
    """
    The spatial distance between each similar pixel and the pixel to be filled is calculated
    """
    similar_SD = np.zeros(shape=(len(similar_indexs)))
    for similar_index_num, similar_index in enumerate(similar_indexs):
        similar_SD[similar_index_num] = np.sqrt(np.power((similar_index[0] - cloud_index[0]),2) +
                                                np.power((similar_index[1] - cloud_index[1]),2))
    return similar_SD

@jit(nopython=True)
def Cal_TRMSD(similar_indexs,Target_data,Input_File_Collection,cloud_index):
    """
    The RMSE of time sequence between each similar pixel and the pixel to be filled is calculated
    """
    similar_TRMSD = np.zeros(shape=(len(similar_indexs),Target_data.shape[0]))
      
    for band in range(Target_data.shape[0]):
        for similar_index_num, similar_index in enumerate(similar_indexs):

            TRMSD_file = 0
            for i in range(Input_File_Collection.shape[0]):
                TRMSD_constant = np.power((Input_File_Collection[i,band,similar_index[0],similar_index[1]] - Input_File_Collection[i,band,cloud_index[0],cloud_index[1]]),2)
                TRMSD_file += TRMSD_constant
            
            similar_TRMSD[similar_index_num,band] = np.sqrt((TRMSD_file / Input_File_Collection.shape[0]))

    return similar_TRMSD

@jit(nopython=True)
def Cal_Weight(TRMSD, SD):
    """
    The weights of each similar pixel in different bands are calculated
    """   
    weight = np.zeros(shape=(TRMSD.shape))
    norm_SD = _norm(SD)

    for i in range(TRMSD.shape[1]):
        if np.any(TRMSD[:,i] == 0):
            index0 = np.argwhere(TRMSD[:,i] == 0)
            for ind in index0:
                weight[ind,i] = 1/len(index0)
        else:
            norm_TRMSD = _norm(TRMSD[:,i])
            D = (norm_SD + 1 ) * (norm_TRMSD + 1)
            D_inverse = np.divide(np.ones_like(D)*1.0, D)
            weight[:,i] = D_inverse / np.sum(D_inverse)

    return weight

@jit(nopython=True)
def Cal_NTL_SP(Target_data,similar_indexs,similar_weight):
    """
    Calculate the spatial prediction value
    """
    NTL_SP = np.zeros(shape=(Target_data.shape[0]))
    for band in range(Target_data.shape[0]):
        for ind,similar_index in enumerate(similar_indexs):
            NTL_SP[band] += Target_data[band,similar_index[0],similar_index[1]] * similar_weight[ind,band]
    
    return NTL_SP

@jit(nopython=True)
def Cal_SDRD(similar_indexs,Input_File_Collection,Target_data):
    """
    Calculate the standard deviation of DN difference for each image
    """
    input_SDRD = np.zeros(shape=(Input_File_Collection.shape[0],Target_data.shape[0]))

    for band in range(Target_data.shape[0]):
        for i in range(Input_File_Collection.shape[0]):
            RD_array = np.zeros(shape=(similar_indexs.shape[0]))

            for ind,similar_index in enumerate(similar_indexs):
                RD_array[ind] = Target_data[band,similar_index[0],similar_index[1]] - Input_File_Collection[i,band,similar_index[0],similar_index[1]] 
            input_SDRD[i,band] = np.std(RD_array)
    
    return input_SDRD

def Cal_TD(Input_date_list,Target_date):
    """
    Calculate the time distance between two dates
    """
    TD = np.zeros(shape=(len(Input_date_list)))
    for i in range(len(Input_date_list)):
        TD[i] = np.abs(_day_interval(Target_date, Input_date_list[i]))

    return TD

@jit(nopython=True)
def Cal_Reliability(SDRD, TD):
    """
    Calculate the reliability of each image
    """
    Reliability = np.zeros(shape=(SDRD.shape))
    norm_TD = _norm(TD)

    for i in range(SDRD.shape[1]):
        if np.any(SDRD[:,i] == 0):
            index0 = np.argwhere(SDRD[:,i] == 0)
            for ind in index0:
                Reliability[ind,i] = 1/len(index0)
        else:

            norm_SDRD = _norm(SDRD[:,i])
            D = (norm_TD + 1 ) * (norm_SDRD + 1)
            D_inverse = np.divide(np.ones_like(D)*1.0, D)
            Reliability[:,i] = D_inverse / np.sum(D_inverse)
  
    return Reliability

@jit(nopython=True)
def Cal_NTL_TP(Reliability,similar_indexs,Target_data,similar_weight,Input_File_Collection,cloud_index):
    """
    Calculate the time predictor results
    """
    NTL_TP = np.zeros(shape=(Reliability.shape))
    for band in range(Reliability.shape[1]):
        for i in range(Input_File_Collection.shape[0]):
            RD_Weight_sum = 0
            for ind,similar_index in enumerate(similar_indexs):
                RD_Weight_sum += similar_weight[ind,band] * (Target_data[band,similar_index[0],similar_index[1]] - Input_File_Collection[i,band,similar_index[0],similar_index[1]]) 
            NTL_TP[i,band] = Input_File_Collection[i,band,cloud_index[0],cloud_index[1]] + RD_Weight_sum

    NTL_TP_final = np.zeros(shape=(NTL_TP.shape[1]))

    for band in range(NTL_TP.shape[1]):
        for j in range(NTL_TP.shape[0]):
            NTL_TP_final[band] += NTL_TP[j,band] * Reliability[j,band]

    return NTL_TP_final

@jit(nopython=True,parallel=True)
def Cal_NTL_Predict(NTL_SP,NTL_TP,TRMSD,SDRD):
    """
    Synthesis of spatiotemporal predicted values
    """
    NTL_Predict = np.zeros(shape=(NTL_SP.shape))
    for band in prange(NTL_TP.shape[0]):

        norm_TRMSD = _norm(TRMSD[:,band])
        norm_SDRD = _norm(SDRD[:,band])

        Uncertainty_S = np.mean(norm_TRMSD)
        Uncertainty_T = np.mean(norm_SDRD)

        Weight_S = Uncertainty_T / (Uncertainty_S + Uncertainty_T)
        Weight_T = Uncertainty_S / (Uncertainty_S + Uncertainty_T)

        NTL_Predict[band] = Weight_S * NTL_SP[band] + Weight_T * NTL_TP[band]

    return NTL_Predict

@jit(nopython=True)
def _Search_Same_Type(cloud_index,Annual_GMM_data,Temporals_GMM_data,cloud_data):
    """
    Search for same type pixels
    """
    Annual_cloud_type = Annual_GMM_data[cloud_index[0],cloud_index[1]]
    Temporals_cloud_type = Temporals_GMM_data[cloud_index[0],cloud_index[1]]

    cloud_data_0 = 1 - cloud_data
    GMM_data_without_cloud = Annual_GMM_data * (Temporals_GMM_data + 100) * cloud_data_0
    same_type_indexs = np.argwhere(GMM_data_without_cloud == (Annual_cloud_type * (Temporals_cloud_type + 100)))

    return same_type_indexs

def _Search_Similar(same_GMM_type_indexs, Annual_data, cloud_index, Ref_pixel_num):
    """
    The RMSD value is calculated to find similar pixels
    """
    same_type_RMSD = _cal_RMSD(same_GMM_type_indexs,Annual_data,cloud_index)
    similar_TRMSD_index = np.argpartition(same_type_RMSD, Ref_pixel_num)[:Ref_pixel_num]
    similar_indexs = same_GMM_type_indexs[similar_TRMSD_index,:]

    return similar_indexs

@jit(nopython=True)
def _Search_SONDI_similar(Input_spectrum_Collection,cloud_index,similar_indexs,Target_spectrum_data,std_per):
    """
    Search for pixels of similar light source types
    """
    cloud_spectrum_list = np.zeros(shape=(Input_spectrum_Collection.shape[0]))
    for i in range(Input_spectrum_Collection.shape[0]):
        cloud_spectrum_list[i] = Input_spectrum_Collection[i,cloud_index[0],cloud_index[1]]

    cloud_spectrum_mean = np.mean(cloud_spectrum_list)
    cloud_spectrum_std = np.std(cloud_spectrum_list)

    spectrum_similar_index_list = []
    for similar_index in similar_indexs:
        if cloud_spectrum_mean - std_per * cloud_spectrum_std < Target_spectrum_data[similar_index[0],similar_index[1]]<cloud_spectrum_mean + std_per * cloud_spectrum_std :
            spectrum_similar_index_list.append(similar_index)

    return spectrum_similar_index_list

@jit(nopython=True)
def _calculate_rmse(predictions, targets):
    """
    Calculates the RMSE values of both arrays
    """
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    return rmse

@jit(nopython=True)
def _cal_RMSD(same_type_indexs,Annual_data,cloud_index):
    """
    Calculate the RMSD value for each pixel
    """
    same_type_RMSD = np.zeros(shape=(len(same_type_indexs)))
    for same_type_num,same_type_index in enumerate(same_type_indexs):

        same_type_value =  np.zeros(shape=(Annual_data.shape[0]))
        center_value =  np.zeros(shape=(Annual_data.shape[0]))
        
        for band in range(Annual_data.shape[0]):

            same_type_value[band] = Annual_data[band,same_type_index[0],same_type_index[1]]
            center_value[band] = Annual_data[band,cloud_index[0],cloud_index[1]]

        same_type_RMSD[same_type_num] = _calculate_rmse(same_type_value, center_value)
    return same_type_RMSD

@jit(nopython=True)
def _norm(list):
    """
    Array maximum and minimum scaling
    """
    norm_list = np.zeros(shape=(list.shape))

    list_max = np.max(list)
    list_min = np.min(list)

    original_range = np.array([list_min, list_max])
    target_range = np.array([0, 1])
    for k in range(len(list)):
        norm_list[k] =  np.interp(list[k], original_range, target_range)

    return norm_list

def _day_interval(day1, day2):
    """
    Calculate the time interval
    """

    time_array1 = time.strptime(day1, "%Y%m%d")
    timestamp_day1 = int(time.mktime(time_array1))

    time_array2 = time.strptime(day2, "%Y%m%d")
    timestamp_day2 = int(time.mktime(time_array2))

    result = (timestamp_day2 - timestamp_day1) // 60 // 60 // 24
    
    return result