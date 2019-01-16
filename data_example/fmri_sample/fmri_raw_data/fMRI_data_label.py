#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : fMRI_data_label.py
# @Author: MoonKuma
# @Date  : 2019/1/16
# @Desc  : label fMRI data


import os
import shutil

Beta_A1 = [
    'beta_0001.nii',
    'beta_0003.nii',
    'beta_0005.nii',
    'beta_0007.nii',
    'beta_0009.nii',
    'beta_0011.nii',
    'beta_0013.nii',
    'beta_0015.nii',
    'beta_0017.nii',
    'beta_0019.nii',
    'beta_0021.nii',
    'beta_0023.nii',
    'beta_0025.nii',
    'beta_0027.nii',
    'beta_0029.nii',
    'beta_0031.nii',
    'beta_0033.nii',
    'beta_0035.nii',
    'beta_0045.nii',
    'beta_0047.nii',
    'beta_0049.nii',
    'beta_0051.nii',
    'beta_0053.nii',
    'beta_0055.nii',
    'beta_0057.nii',
    'beta_0059.nii',
    'beta_0061.nii',
    'beta_0063.nii',
    'beta_0065.nii',
    'beta_0067.nii',
    'beta_0069.nii',
    'beta_0071.nii',
    'beta_0073.nii',
    'beta_0075.nii',
    'beta_0077.nii',
    'beta_0079.nii',
    'beta_0089.nii'	,
    'beta_0091.nii',
    'beta_0093.nii',
    'beta_0095.nii',
    'beta_0097.nii',
    'beta_0099.nii',
    'beta_0101.nii',
    'beta_0103.nii',
    'beta_0105.nii',
    'beta_0107.nii',
    'beta_0109.nii',
    'beta_0111.nii',
    'beta_0113.nii',
    'beta_0115.nii',
    'beta_0117.nii',
    'beta_0119.nii',
    'beta_0121.nii',
    'beta_0123.nii',
    'beta_0133.nii',
    'beta_0135.nii',
    'beta_0137.nii',
    'beta_0139.nii',
    'beta_0141.nii',
    'beta_0143.nii',
    'beta_0145.nii',
    'beta_0147.nii',
    'beta_0149.nii',
    'beta_0151.nii',
    'beta_0153.nii',
    'beta_0155.nii',
    'beta_0157.nii',
    'beta_0159.nii',
    'beta_0161.nii',
    'beta_0163.nii',
    'beta_0165.nii',
    'beta_0167.nii'
]

'''
CAF(:,:,:,1)=CAF(:,:,:,1)+(NewImg_persub(:,:,:,(i-1)*18+7)+NewImg_persub(:,:,:,(i-1)*18+15))/2;
CAM(:,:,:,1)=CAM(:,:,:,1)+(NewImg_persub(:,:,:,(i-1)*18+8)+NewImg_persub(:,:,:,(i-1)*18+16))/2;
CHF(:,:,:,1)=CHF(:,:,:,1)+(NewImg_persub(:,:,:,(i-1)*18+9)+NewImg_persub(:,:,:,(i-1)*18+11))/2;
CHM(:,:,:,1)=CHM(:,:,:,1)+(NewImg_persub(:,:,:,(i-1)*18+10)+NewImg_persub(:,:,:,(i-1)*18+12))/2;
AFF(:,:,:,1)=AFF(:,:,:,1)+(NewImg_persub(:,:,:,(i-1)*18+13)+NewImg_persub(:,:,:,(i-1)*18+17))/2;
AFM(:,:,:,1)=AFM(:,:,:,1)+(NewImg_persub(:,:,:,(i-1)*18+14)+NewImg_persub(:,:,:,(i-1)*18+18))/2;
'''

label_map = dict()

for i in range(0, 4):
    label_map[Beta_A1[i * 18 + 7 - 1]] = 'CA_F'
    label_map[Beta_A1[i * 18 + 15 - 1]] = 'CA_F'
    label_map[Beta_A1[i * 18 + 8 - 1]] = 'CA_M'
    label_map[Beta_A1[i * 18 + 16 - 1]] = 'CA_M'
    label_map[Beta_A1[i * 18 + 9 - 1]] = 'CH_F'
    label_map[Beta_A1[i * 18 + 11 - 1]] = 'CH_F'

    label_map[Beta_A1[i * 18 + 10 - 1]] = 'CH_M'
    label_map[Beta_A1[i * 18 + 12 - 1]] = 'CH_M'
    label_map[Beta_A1[i * 18 + 13 - 1]] = 'AF_F'
    label_map[Beta_A1[i * 18 + 17 - 1]] = 'AF_F'
    label_map[Beta_A1[i * 18 + 14 - 1]] = 'AF_M'
    label_map[Beta_A1[i * 18 + 18 - 1]] = 'AF_M'

print(label_map)

sub_list = [5]
data_path = 'C:/Users/7q/PycharmProjects/tf_fmri_meg/data_example/fmri_sample/fmri_raw_data/sub'
new_local = 'C:/Users/7q/PycharmProjects/tf_fmri_meg/data_example/fmri_sample/fmri_data'
for sub_num in sub_list:
    data_path_sub = data_path + str(sub_num)
    file_list = os.listdir(data_path_sub)
    for file_name in file_list:
        if file_name in label_map.keys():
            new_name = new_local + '/' + label_map[file_name] + '/' + os.path.splitext(file_name)[0] + '_sub_' + str(sub_num) + os.path.splitext(file_name)[1]
            old_name = data_path_sub + '/' + file_name
            shutil.copyfile(old_name, new_name)


