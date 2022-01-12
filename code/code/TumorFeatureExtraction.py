# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:44:09 2019

@author: PC
"""

import xlrd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor,imageoperations
import os
import pandas as pd
from pandas import DataFrame as DF
import warnings
import time
from time import sleep
from tqdm import tqdm
from skimage import measure



    
def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

def Extract_Features(image,mask,params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    data_path = r"F:\HGSOCTumor\ChemotherapeuticSensitivity\ImgData"
    workbook = xlrd.open_workbook(os.path.join(data_path,'铂耐药名单.xlsx'))
    sheet_names = workbook.sheet_names()
    for i in range(len(sheet_names)-1):
        sheet_object = workbook.sheet_by_name(sheet_names[i])
        patient_name = sheet_object.col_values(0)
        age = sheet_object.col_values(2)
        resistance = sheet_object.col_values(10)
        PFS_time = sheet_object.col_values(12)
        OS_status = sheet_object.col_values(13)
        OS_time = sheet_object.col_values(14)
        stage = sheet_object.col_values(16)
        residual = sheet_object.col_values(17)
        mass_characteristic = sheet_object.col_values(19)
        bilaterality = sheet_object.col_values(21)
        CA125 = sheet_object.col_values(23)
        Img_Path = os.path.join(data_path,sheet_names[i])
        img_list = os.listdir(Img_Path)
        PDS_file = [file for file in img_list if file[:3]=='PDS']
        PDS_path = os.path.join(Img_Path, PDS_file[0])
        result_path = './Result'
        save_path = os.path.join(result_path,sheet_names[i])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        T1_Feature = []
        T2_Feature = []
        for j in range(1,len(patient_name)):
            patient_path = os.path.join(PDS_path,patient_name[j].strip())
            img_filenames = os.listdir(patient_path)
            T1_folder = [i for i in img_filenames if i[:2]=='T1' and len(i.split('.'))==1]
            T2_folder = [i for i in img_filenames if i[:2]=='T2' and len(i.split('.'))==1]
            T1_maskpath = [i for i in img_filenames if i[:2]=='T1' and len(i.split('.'))==2]
            T2_maskpath = [i for i in img_filenames if i[:2]=='T2' and len(i.split('.'))==2]
            if len(T1_maskpath)==2:
                T1_maskpath = [i for i in T1_maskpath if len(i.split('-'))==2]
            if len(T2_maskpath)==2:
                T2_maskpath = [i for i in T2_maskpath if len(i.split('-'))==2]
            T1_img = readDCM_Img(os.path.join(patient_path, T1_folder[0]))
            T1_mask = sitk.ReadImage(os.path.join(patient_path, T1_maskpath[0]))
            T2_img = readDCM_Img(os.path.join(patient_path, T2_folder[0]))
            T2_mask = sitk.ReadImage(os.path.join(patient_path, T2_maskpath[0]))
            T1_features, T1_feature_info = Extract_Features(T1_img, T1_mask, 'params_T1.yaml')
            T2_features, T2_feature_info = Extract_Features(T2_img, T2_mask, 'params_T2.yaml')
            T1_features['Name'] = patient_name[j]
            T1_features['resistance'] = resistance[j]
            T1_features['PFS_time'] = PFS_time[j]
            T1_features['OS_status'] = OS_status[j]
            T1_features['OS_time'] = OS_time[j]
            T1_features['age'] = age[j]
            T1_features['stage'] = stage[j]
            T1_features['residual'] = residual[j]
            T1_features['mass_characteristic'] = mass_characteristic[j]
            T1_features['bilaterality'] = bilaterality[j]
            T1_features['CA125'] = CA125[j]
            T1_Feature.append(T1_features)
            
            T2_features['Name'] = patient_name[j]
            T2_features['resistance'] = resistance[j]
            T2_features['PFS_time'] = PFS_time[j]
            T2_features['OS_status'] = OS_status[j]
            T2_features['OS_time'] = OS_time[j]
            T2_features['age'] = age[j]
            T2_features['stage'] = stage[j]
            T2_features['residual'] = residual[j]
            T2_features['mass_characteristic'] = mass_characteristic[j]
            T2_features['bilaterality'] = bilaterality[j]
            T2_features['CA125'] = CA125[j]
            T2_Feature.append(T2_features)
        df = DF(T2_Feature).fillna('0')
        df.to_csv(os.path.join(save_path,'T2_Feature.csv'),index=False,sep=',')
        df = DF(T1_Feature).fillna('0')
        df.to_csv(os.path.join(save_path,'T1_Feature.csv'),index=False,sep=',')
