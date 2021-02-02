import copy
import json
import logging
import numpy as np
import os
from pathlib import Path
import platform
import sys
from .artificial_generation_setting import load_deform_exp_setting
from .experiment_setting import load_global_step_from_predefined_list
from .experiment_setting import fancy_exp_name
from .experiment_setting import clean_exp_name


def initialize_setting(current_experiment, where_to_run=None):
    if where_to_run is None:
        where_to_run = 'Auto'
    setting = dict()
    setting['where_to_run'] = where_to_run
    setting['RootFolder'], setting['log_root_folder'], setting['DataFolder'] = root_address_generator(where_to_run=where_to_run)
    setting['current_experiment'] = current_experiment
    setting['stage'] = 1
    setting['UseLungMask'] = True          # The peaks of synthetic deformation can only be inside the mask
    setting['UseTorsoMask'] = True         # set background region to setting[DefaultPixelValue]
    setting['verbose_image'] = False       # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['WriteDVFStatistics'] = False
    setting['ParallelSearching'] = True
    setting['DVFPad_S1'] = 0
    setting['DVFPad_S2'] = 0
    setting['DVFPad_S4'] = 0
    setting['VoxelSize'] = [2, 2, 2.5] 
    setting['data'] = dict()
    setting['DataList'] = ['SPREAD']
    setting['data']['SPREAD'] = load_data_setting('SPREAD')
    setting['Dim'] = 3   # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    setting['Augmentation'] = False
    setting['WriteBSplineTransform'] = False
    setting['verbose'] = True           # Detailed printing
    setting['normalization'] = None     # The method to normalize the intensities: 'linear'
    return setting


def root_address_generator(where_to_run='Auto'):
    """
    choose the root folder, you can modify the addresses:
        'Auto'
        'Cluster'
        'Root'
    :param where_to_run:
    :return:
    """
    if where_to_run == 'Root':
        root_folder = './'
        data_folder = './Data/'
    elif where_to_run == 'Auto':
        print(sys.platform)
        if sys.platform == 'win32':
            root_folder = 'E:/PHD/Software/Project/DL/'
            data_folder = 'E:/PHD/Database/'
        elif sys.platform == 'linux':
            root_folder = './data/'
            data_folder = './data/'
        else:
            raise ValueError('sys.platform is only defined in ["win32"]. Please defined new os in setting.setting_utils.root_address_generator()')
    elif where_to_run == 'Cluster':
        root_folder = '/exports/lkeb-hpc/hsokootioskooyi/Project/DL/'
        data_folder = '/exports/lkeb-hpc/hsokootioskooyi/Data/'
    else:
        raise ValueError('where_to_run is only defined in ["Root", "Auto", "Cluster"]. Please defined new os in setting.setting_utils.root_address_generator()')
    log_root_folder = root_folder + 'TB/'
    return root_folder, log_root_folder, data_folder


def load_setting_from_data_dict(setting, data_exp_dict_list):
    """
    :param setting:
    :param data_exp_dict_list:

        Load two predefined information:
        1. load the general setting of selected data with load_data_setting(selected_data)
        2. load the all settings of the deform_exp with load_deform_exp_setting(selected_deform_exp)

        Two list are also updated in order to have redundant information setting['DataList'], setting['DeformExpList']
    :return: setting
    """
    setting['DataExpDict'] = data_exp_dict_list
    setting['data'] = dict()
    setting['deform_exp'] = dict()
    data_list = []
    deform_exp_list = []
    for data_exp_dict in data_exp_dict_list:
        data_list.append(data_exp_dict['data'])
        setting['data'][data_exp_dict['data']] = load_data_setting(data_exp_dict['data'])
        if 'deform_exp' in data_exp_dict.keys():
            deform_exp_list.append(data_exp_dict['deform_exp'])
            setting['deform_exp'][data_exp_dict['deform_exp']] = load_deform_exp_setting(data_exp_dict['deform_exp'])
    setting['DataList'] = data_list
    setting['DeformExpList'] = deform_exp_list
    return setting


def load_data_setting(selected_data):
    """
    load the general setting of selected data like default pixel value and types of images (baseline, follow-up...)
    :param selected_data:
    :return:
    """
    data_setting = dict()
    if selected_data == 'SPREAD':
        data_setting['ext'] = '.mha'
        data_setting['ImageByte'] = 2                # equals to sitk.sitkInt16 , we prefer not to import sitk in setting_utils
        data_setting['types'] = ['Fixed', 'Moving']  # for eg: 'Fixed' or 'Moving' : actually Fixed indicates baseline and Moving indicates followup
        data_setting['expPrefix'] = 'ExpLung'        # for eg: ExpLung
        data_setting['DefaultPixelValue'] = -2048    # The pixel value when a transformed pixel is outside of the image
        data_setting['VoxelSize'] = [1, 1, 1]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = True
        data_setting['CNList'] = [i for i in range(1, 21)]

    elif selected_data == 'DIR-Lab_4D':
        data_setting['ext'] = '.mha'
        data_setting['ImageByte'] = 2              # equals to sitk.sitkInt16 , we prefer not to import sitk in setting_utils
        data_setting['types'] = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']     # for eg: 'Fixed' or 'Moving'
        data_setting['expPrefix'] = 'case'         # for eg: case
        data_setting['DefaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
        data_setting['VoxelSize'] = [2, 2, 2.5]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = False
        data_setting['CNList'] = [i for i in range(6, 11)]

    elif selected_data == 'DIR-Lab_COPD':
        data_setting['ext'] = '.mha'
        data_setting['ImageByte'] = 2                   # equals to sitk.sitkInt16 , we prefer not to import sitk in setting_utils
        data_setting['types'] = ['iBHCT', 'eBHCT']      # for eg: 'Fixed' or 'Moving'
        data_setting['expPrefix'] = 'copd'              # for eg: case
        data_setting['DefaultPixelValue'] = -2048       # The pixel value when a transformed pixel is outside of the image
        data_setting['VoxelSize'] = [1, 1, 1]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = False
        data_setting['CNList'] = [i for i in range(1, 11)]
    else:
        logging.warning('warning: -------- selected_data not found')
    return data_setting


def address_generator(s, requested_address, data=None, deform_exp=None, type_im=0, cn=1, dsmooth=0, print_mode=False,
                      dvf_pad=None, stage=None, stage_list=None, train_mode='', c=0, semi_epoch=0, chunk=0,
                      root_folder=None, plot_mode=None, plot_itr=None, plot_i=None, current_experiment=None,
                      step=None, pair_info=None, deformed_im_ext=None, im_list_info=None, ishuffled_exp=None,
                      padto=None, bspline_folder=None, spacing=None, dvf_threshold_list=None, base_reg=None):
    if data is None:
        data = s['DataList'][0]
    if deform_exp is None:
        deform_exp = ''
        if len(s.get('DeformExpList', [])) > 0:
            deform_exp = s['DeformExpList'][0]
    if current_experiment is None:
        current_experiment = s.get('current_experiment', None)
    if '/' in current_experiment:
        log_sub_folder = current_experiment.rsplit('/')[0]
        current_experiment = current_experiment.rsplit('/')[1]
    else:
        log_sub_folder = 'RegNet'
    if root_folder is None:
        root_folder = s.get('RootFolder', None)
    deform_exp_folder = root_folder+'Elastix/Artificial_Generation/'+deform_exp+'/'+data+'/'

    if stage is None:
        stage = s.get('stage', None)
    if stage_list is None:
        if 'stage_list' in s.keys():
            stage_list = s['stage_list']

    read_pair_mode = s.get('read_pair_mode', 'real')
    if base_reg is None:
        base_reg = s.get('BaseReg', '')

    # if read_pair_mode == 'synthetic':
    #     if requested_address == 'MovedIm':
    #         requested_address = 'MovedIm_AG'

    if dvf_pad is None:
        dvf_pad = s.get('DVFPad_S' + str(stage), None)

    if bspline_folder is None:
        bspline_folder = s.get('Reg_BSpline_Folder', None)

    exp_prefix = s['data'][data]['expPrefix'] + str(cn)

    if requested_address in ['Im', 'Torso', 'Lung']:
        if dsmooth == 0:
            requested_address = 'Original' + requested_address
            if spacing == 'Raw':
                requested_address = requested_address + 'Raw'

        elif dsmooth > 0:
            requested_address = 'Next' + requested_address
    ext = s['data'][data]['ext']
    type_im_name = s['data'][data]['types'][type_im]
    address = {}
    name_dic = {}
    if data == 'SPREAD':
        if requested_address in ['OriginalFolder', 'OriginalIm', 'OriginalLung', 'OriginalTorso', 'DilatedLandmarksIm']:
            name_dic['OriginalIm'] = type_im_name + 'ImageFullRS1'
            name_dic['OriginalLung'] = type_im_name + 'MaskFullRS1'
            name_dic['OriginalTorso'] = type_im_name + 'TorsoFullRS1'
            name_dic['DilatedLandmarksIm'] = 'DilatedLandmarksFullRS1'
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            address['OriginalFolder'] = root_folder + 'Elastix/LungExp/' + exp_prefix + '/Result/'
            if requested_address != 'OriginalFolder':
                address[requested_address] = address['OriginalFolder'] + name_dic[requested_address] + ext

        elif requested_address in ['OriginalImNonIsotropic', 'OriginalLandmarkFolder', 'LandmarkIndex_tr', 'LandmarkIndex_elx',
                                   'LandmarkPoint_tr', 'LandmarkPoint_elx', 'UnsurePoints']:
            patient_case = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA',
                            'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA',
                            'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
            type_im_landmark_tr = [patient_case[cn-1] + '_baseline_1_Cropped_point_trunc.txt',
                                   'Consensus/' + patient_case[cn-1][0:4] + '_b1f1_point_trunc.txt']
            type_im_landmark_elx = [patient_case[cn-1] + '_baseline_1_Cropped_point.txt',
                                    'Consensus/' + patient_case[cn-1][0:4] + '_b1f1_point.txt']

            address['OriginalLandmarkFolder'] = s['DataFolder'] + 'lung_dataset/SPREAD/SPREADgroundTruth/'
            original_names = ['baseline_1.mhd', 'followup_1.mhd']
            address['OriginalImNonIsotropic'] = s['DataFolder']+'lung_dataset/SPREAD/'+patient_case[cn-1]+'/'+original_names[type_im]
            address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder'] + type_im_landmark_tr[type_im]
            address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder'] + type_im_landmark_elx[type_im]
            address['UnsurePoints'] = address['OriginalLandmarkFolder'] + 'Consensus/' + patient_case[cn - 1][0:4] + '_b1f1_unsure.txt'

    elif data in ['DIR-Lab_4D', 'DIR-Lab_COPD']:
        if data == 'DIR-Lab_4D':
            dir_lab_folder = '4DCT'
        else:
            dir_lab_folder = 'COPDgene'
        if requested_address in ['OriginalImNonIsotropic', 'OriginalImNonIsotropicFolder']:
            address['OriginalImNonIsotropicFolder'] = s['DataFolder'] + 'DIR-Lab/'+dir_lab_folder+'/mha/' + exp_prefix + '/'
            address['OriginalImNonIsotropic'] = address['OriginalImNonIsotropicFolder']+exp_prefix+'_' +\
                type_im_name + ext
        elif requested_address in ['OriginalFolder', 'OriginalIm', 'OriginalImRaw', 'OriginalLung', 'OriginalLungRaw',
                                   'OriginalTorso', 'OriginalTorsoRaw', 'DilatedLandmarksIm', 'DilatedLandmarksImNonIsotropic']:

            address['OriginalFolder'] = s['DataFolder'] + dir_lab_folder+'/mha/'+exp_prefix+'/'
            name_dic['OriginalIm'] = exp_prefix + '_' + type_im_name + '_RS1'
            name_dic['OriginalImRaw'] = exp_prefix + '_' + type_im_name
            name_dic['OriginalLung'] = 'Lung_Filled/' + exp_prefix + '_' + type_im_name + '_Lung_Filled_RS1'
            name_dic['OriginalLungRaw'] = 'Lung_Filled/' + exp_prefix + '_' + type_im_name + '_Lung_Filled'
            name_dic['OriginalTorso'] = 'Torso/' + exp_prefix + '_' + type_im_name + '_Torso_RS1'
            name_dic['OriginalTorsoRaw'] = 'Torso/' + exp_prefix + '_' + type_im_name + '_Torso'
            name_dic['DilatedLandmarksIm'] = 'dilatedLandmarks' + str(cn) + '_' + type_im_name + '_Landmarks_RS1'
            name_dic['DilatedLandmarksImNonIsotropic'] = 'dilatedLandmarks' + str(cn) + '_' + type_im_name
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            if requested_address != 'OriginalFolder':
                address[requested_address] = address['OriginalFolder'] + name_dic[requested_address] + ext

        elif requested_address in ['OriginalLandmarkFolder', 'LandmarkIndex', 'LandmarkIndex_tr', 'LandmarkIndex_elx',
                                   'LandmarkPoint_tr', 'LandmarkPoint_elx']:

            if data == 'DIR-Lab_4D':
                address['OriginalLandmarkFolder'] = s['DataFolder']+dir_lab_folder+'/points/'+exp_prefix+'/'
                if ((pair_info[0]['type_im'] == 0 and pair_info[1]['type_im'] == 5) or
                    (pair_info[0]['type_im'] == 5 and pair_info[1]['type_im'] == 0)):
                        # pair_info[0]['cn'] <= 5 and pair_info[1]['cn'] <= 5:
                    address['LandmarkIndex'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz.txt'
                    address['LandmarkIndex_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz_tr.txt'
                    address['LandmarkIndex_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz_elx.txt'
                    address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_world_tr.txt'
                    address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_world_elx.txt'
                else:
                    address['LandmarkIndex'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz.txt'
                    address['LandmarkIndex_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz_tr.txt'
                    address['LandmarkIndex_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz_elx.txt'
                    address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_world_tr.txt'
                    address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_world_elx.txt'

            if data == 'DIR-Lab_COPD':
                address['OriginalLandmarkFolder'] = s['DataFolder']+'DIR-Lab/'+dir_lab_folder+'/points/'+exp_prefix+'/'
                type_im_landmark = s['data'][data]['types'][type_im][0:3]
                # address['LandmarkIndex'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_name+'_r1_xyz.txt'
                address['LandmarkIndex_tr'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_xyz_r1_tr.txt'
                address['LandmarkIndex_elx'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_xyz_r1_elx.txt'
                address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_world_r1_tr.txt'
                address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_world_r1_elx.txt'
    if requested_address in ['ParameterFolder']:
        address['ParameterFolder'] = root_folder + 'Elastix/Registration/Parameter/'

    if requested_address in ['BaseRegFolder', 'MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg', 'Reg_BaseReg_OutputPoints', 'TransformParameterBaseReg']:

        subfolder_base_reg = pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_'+pair_info[0]['spacing']+'_' +\
                           pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'_'+pair_info[1]['spacing']+'/'
        address['BaseRegFolder'] = root_folder+'Elastix/Registration/'+base_reg+'/'+data+'/'+subfolder_base_reg
        address['Reg_BaseReg_OutputPoints'] = address['BaseRegFolder'] + 'outputpoints.txt'
        address['TransformParameterBaseReg'] = address['BaseRegFolder'] + 'TransformParameters.0.txt'

        if requested_address in ['MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg']:
            name_dic['MovedImBaseReg'] = 'result.0'
            name_dic['MovedTorsoBaseReg'] = 'MovedTorso'+base_reg
            name_dic['MovedLungBaseReg'] = 'MovedLung'+base_reg
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            address[requested_address] = address['BaseRegFolder'] + name_dic[requested_address] + ext

    # if requested_address in ['BaseRegFolder', 'MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg', 'Reg_BaseReg_OutputPoints']:
    #     subfolder_base_reg = pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_'+pair_info[0]['spacing']+'_' +\
    #                        pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'_'+pair_info[1]['spacing']+'/'
    #     address['BaseRegFolder'] = root_folder+'Elastix/Registration/'+base_reg+'/'+data+'/'+subfolder_base_reg
    #     address['Reg_BaseReg_OutputPoints'] = address['BaseRegFolder'] + 'outputpoints.txt'
    #
    #     if requested_address in ['MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg']:
    #         name_dic['MovedImBaseReg'] = 'result.0'
    #         name_dic['MovedTorsoBaseReg'] = 'MovedTorso'+base_reg
    #         name_dic['MovedLungBaseReg'] = 'MovedLung'++base_reg
    #         if stage > 1:
    #             name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
    #         address[requested_address] = address['BaseRegFolder'] + name_dic[requested_address] + ext

    elif requested_address in ['BSplineFolder', 'MovedImBSpline', 'BSplineOutputParameter', 'DVFBSpline', 'DVFBSpline_Jac', 'Reg_BSpline_OutputPoints']:
        subfolder_bspline = pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_'+pair_info[0]['spacing']+'_' +\
                           pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'_'+pair_info[1]['spacing']+'/'
        address['BSplineFolder'] = root_folder+'Elastix/Registration/BSpline/'+data+'/'+bspline_folder+'/'+subfolder_bspline
        address['MovedImBSpline'] = address['BSplineFolder'] + 'result.0' + ext
        address['BSplineOutputParameter'] = address['BSplineFolder'] + 'TransformParameters.0.txt'
        address['DVFBSpline'] = address['BSplineFolder'] + 'deformationField' + ext
        address['DVFBSpline_Jac'] = address['BSplineFolder'] + 'Jac' + ext
        address['Reg_BSpline_OutputPoints'] = address['BSplineFolder'] + 'outputpoints.txt'

    elif requested_address in ['NextFolder', 'NextIm', 'NextLung', 'NextTorso', 'NextDVF', 'NextJac', 'NextBSplineTransform', 'NextBSplineTransformIm']:
        address['NextFolder'] = deform_exp_folder+type_im_name+'/'+exp_prefix+'/Dsmooth0'+'/DNext'+str(dsmooth)+'/'
        if print_mode:
            address['NextFolder'] = deform_exp+'/'+data+'/'+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth0'+'/DNext'+str(dsmooth)+'/'
        if requested_address in ['NextBSplineTransform']:
            address[requested_address] = address['NextFolder'] + 'NextBSplineTransform.tfm'
        if requested_address in ['NextIm', 'NextLung', 'NextTorso', 'NextDVF', 'NextJac', 'NextBSplineTransformIm']:
            name_dic[requested_address] = requested_address
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            address[requested_address] = address['NextFolder'] + name_dic[requested_address] + ext

    elif requested_address in ['DSmoothFolder', 'DFolder', 'DeformedIm', 'DeformedOccluded', 'DeformedDVF', 'DeformedDVFLabel', 'DeformedArea', 'DeformedTorso',
                               'DeformedLung', 'DeformedLungOccluded', 'DVF_histogram', 'Jac', 'Jac_histogram', 'ImCanny', 'BSplineTransform',
                               'BSplineTransformIm', 'MovedIm_AG', 'MovedLung_AG', 'MovedTorso_AG']:
        dsmooth_mod = dsmooth % len(s['deform_exp'][deform_exp]['DeformMethods'])
        deform_number = get_deform_number_from_dsmooth(s, dsmooth, deform_exp=deform_exp)
        address['DSmoothFolder'] = deform_exp_folder+type_im_name+'/'+exp_prefix+'/Dsmooth'+str(dsmooth)+'/'
        address['DFolder'] = address['DSmoothFolder']+s['deform_exp'][deform_exp]['DeformMethods'][dsmooth_mod]+'_'+'D'+str(deform_number)+'/'
        if print_mode:
            address['DSmoothFolder'] = deform_exp+'/'+data+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth'+str(dsmooth)+'/'
            address['DFolder'] = address['DSmoothFolder']+s['deform_exp'][deform_exp]['DeformMethods'][dsmooth_mod]+'_'+'D'+str(deform_number)+'/'

        if requested_address in ['BSplineTransform']:
            address[requested_address] = address['DFolder'] + 'BSplineTransform.tfm'

        if requested_address in ['DeformedIm', 'DeformedOccluded', 'DeformedDVF', 'DeformedDVFLabel', 'DeformedArea', 'DeformedTorso', 'DeformedLung', 'DeformedLungOccluded',
                                 'ImCanny', 'BSplineTransformIm', 'MovedIm_AG', 'MovedLung_AG', 'MovedTorso_AG']:
            if requested_address == 'ImCanny':
                name_dic['ImCanny'] = 'canny' + str(s['deform_exp'][deform_exp]['Canny_LowerThreshold']) + '_' + str(s['deform_exp'][deform_exp]['Canny_UpperThreshold'])
            elif requested_address in ['DeformedIm']:
                if deformed_im_ext is not None:
                    if isinstance(deformed_im_ext, list):
                        deformed_im_ext_string = deformed_im_ext[0]
                        if len(deformed_im_ext) > 1:
                            for deform_exp_i in deformed_im_ext[1:]:
                                deformed_im_ext_string = deformed_im_ext_string + '_' + deform_exp_i
                    else:
                        deformed_im_ext_string = deformed_im_ext
                    name_dic['DeformedIm'] = 'DeformedImage_' + deformed_im_ext_string
                else:
                    name_dic['DeformedIm'] = 'DeformedImage'
            elif requested_address in ['DeformedDVF', 'DeformedDVFLabel']:
                if requested_address == 'DeformedDVFLabel':
                    name_dic['DeformedDVFLabel'] = 'DeformedDVFLabel'
                    for dvf_threshold in dvf_threshold_list:
                        name_dic['DeformedDVFLabel'] = name_dic['DeformedDVFLabel']+'_'+str(dvf_threshold)
                elif requested_address == 'DeformedDVF':
                    name_dic['DeformedDVF'] = 'DeformedDVF'
                name_dic[requested_address] = name_dic[requested_address] + '_pad' + str(dvf_pad)
            else:
                name_dic[requested_address] = requested_address
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            elif requested_address in ['MovedIm_AG', 'MovedLung_AG', 'MovedTorso_AG']:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            address[requested_address] = address['DFolder'] + name_dic[requested_address] + ext

        if requested_address in ['DVF_histogram', 'Jac', 'Jac_histogram']:
            name_dic[requested_address] = requested_address + '_pad' + str(dvf_pad)
            if stage > 1:
                name_