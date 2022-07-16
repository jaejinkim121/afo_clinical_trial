# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 09:00:29 2022

@author: mleem
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from scipy.signal import find_peaks
import statistics
import math
import operator
from time import time
import joblib

from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler, label_binarize, PolynomialFeatures
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from include.calib_functions import folder_path_name


force_range = np.arange(5.0,55.0,5.0)
RANDOM_STATE = 42
TEST_SIZE = 0.35
N_FOLDS = 10

FEATURES = 7
CASE = 20

base_path = "D:/OneDrive - SNU/AFO_analysis/" + \
    "afo_clinical_trial/data/analyzed/sole/"

(exp_list, exp_name_list) = folder_path_name(base_path, "start", "RH", 1)

for (RH_list, RH_name) in zip(exp_list, exp_name_list):

    # RH-##
    RH_num = RH_name[-2:]

    df_train = pd.DataFrame(columns=[
        "case_num", "RH", "folder_name", "sample_num", "time", "pred_time",
        "explained_variance_score", "mean_absolute_error",
        "mean_squared_error", "RMSE_KPA", "R2"])
    # df_test = pd.DataFrame(columns=["case_num", "RH", "stance_or_walk", "stance_or_walk_num", "sensor_dir", "sensor_num", "sample_num","pred_time"])

    RH_path = str(RH_list) + "/"

    if not os.path.exists(str(RH_path) + "Calibration"):
        calib_case = ["Calibration_post", "Calibration_pre"]
    else:
        calib_case = ["Calibration"]

    if len(calib_case) == 1:
        calib_name = "Calibration"
    else:
        calib_name = "Calibration_post"

    # calibration data path
    calib_path = RH_path + calib_name + "/"
    (calib_test_list, calib_name_list) = folder_path_name(calib_path)
    for (calib_num_path, calib_num_name) in zip(calib_test_list,
                                                calib_name_list):

        for case in np.arange(CASE, CASE + 1, 1):

            print("start_RH%s_%s_CASE%s" %(RH_num, calib_num_name, str(case)))

            if case == 1:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5","elapsed_time1"]
            elif case == 2:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]
            elif case == 3:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6"]
            elif case == 4:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7"]
            elif case == 5:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8"]
            elif case == 6:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9"]
            elif case == 7:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9", "pre_10"]
            elif case == 8:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5","elapsed_time2"]
            elif case == 9:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"] 
            elif case == 10:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6"]
            elif case == 11:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7"]
            elif case == 12:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8"]
            elif case == 13:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9"]
            elif case == 14:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9", "pre_10"]
            elif case == 15:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9", "pre_10","elapsed_time2"]
            elif case == 16:
                X_list = ["vout", "loading_type1", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9", "pre_10","elapsed_time2"]
            elif case == 17:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9", "pre_10","elapsed_time2"]
            elif case == 18:
                X_list = ["vout", "loading_type2", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5","elapsed_time2"]
            elif case == 19:
                X_list = ["vout", "loading_type1", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "pre_6", "pre_7", "pre_8", "pre_9", "pre_10","elapsed_time1"]
            elif case == 20:
                X_list = ["vout"]

            sensor_name = calib_num_name[3:]
            sensor_dir = calib_num_name[3:-2]
            sensor_num = calib_num_name[-1]

            if calib_name == "Calibration":
            
                training_path = str(calib_num_path)+"/force_conversion_test.csv"
                
                ##############################################################
                ########        Training                      ################
                ##############################################################      
                data_path = training_path
                raw_data = pd.read_csv(data_path, header = 0)
                data = raw_data.astype(float)
                
            elif calib_name == "Calibration_post":
                
                training_path1 = str(calib_num_path)+"/force_conversion_test.csv"
                training_path2 = str(RH_path)+"Calibration_pre/"+str(calib_num_name)+"/force_conversion_test.csv"
                
                try: 
                    raw_data2 = pd.read_csv(training_path2, header = 0)
                except FileNotFoundError:
                    data_path = training_path1
                    raw_data = pd.read_csv(data_path, header = 0)
                    data = raw_data.astype(float)
                    pass
                else:
                    raw_data1 = pd.read_csv(training_path1, header = 0)
                    data1 = raw_data1.astype(float)
                    
                    raw_data2 = pd.read_csv(training_path2, header = 0)
                    data2 = raw_data2.astype(float)
                    
                    data = pd.concat([data1,data2],axis=0)
                    data.reset_index(drop=True, inplace=True)
                    
                    pass
            # force class for uniform sampling
            data["force_class"] = pd.cut(data.force, bins = np.arange(0.0,55.0,5.0), labels = [str(s) for s in force_range])
            
            # 결측치 제거
            reg_data = data.dropna(axis=0)
            reg_data.reset_index(drop=True, inplace=True)
            
            # class별 minimum number of samples
            NUM_MIN_SAMPLE = int((reg_data["force_class"].value_counts().min()))
            
            if NUM_MIN_SAMPLE > 3200:
                
                NUM_MIN_SAMPLE = 3200
            
            # class별 최소 sample 수만큼만 추출
            reg_data = reg_data.groupby("force_class").apply(lambda x: x.sample(NUM_MIN_SAMPLE, random_state = RANDOM_STATE))
            
            # scaler 적용
            # reg_data[["vout", "force"]] = StandardScaler().fit_transform(reg_data[["vout", "force"]])
            
            # case 1
            # X = reg_data[["vout", "load_unload", "cycle_num", "disp_rate", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "indenter_pos"]]
            # case 2
            # X = reg_data[["vout", "load_unload", "cycle_num", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
            # X = reg_data[["vout", "loading_type", "cycle_num", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
            # case 3
            # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
            X = reg_data[X_list]
            # case 4
            # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1", "pre_2", "pre_3"]]
            # case 5
            # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1"]]
            # case 6
            # X = reg_data[["vout", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
            # case 7
            # X = reg_data[["vout"]].values.reshape(-1,1)
            
            y = reg_data[["force"]].values.reshape(-1,1)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, shuffle = True, stratify = reg_data.force_class, random_state = RANDOM_STATE)
    
            ###############################################################
            # N-dimensional GPR
            ##############################################################
            start_time = time()
            
            gaussian_process = GaussianProcessRegressor(random_state=RANDOM_STATE)
            
            gaussian_process.fit(X_train, y_train)
            # save GPR model
            model_name = "GPR_CASE%s_%s.sav" %(CASE, calib_num_name[3:])
            model_path = base_path + "RH-%s/model/" % RH_num + model_name
            try:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
            except OSError:
                pass
            joblib.dump(gaussian_process, model_path)
            
            end_time = time()
            pred_start_time = time()
            
            mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)
            
            pred_end_time = time()
            # plt.scatter(X['vout'], y, linestyle="dotted")
            # plt.scatter(X_train['vout'], y_train, label="Observations")
            
            # # sort the values of X_test
            # sort_axis = operator.itemgetter(0)
            # sorted_zip = sorted(zip(X_test['vout'], mean_prediction), key=sort_axis)
            # X_test['vout'], mean_prediction = zip(*sorted_zip)
            
            # plt.plot(X_test['vout'], mean_prediction, label="Mean prediction")
    
            ########################################################################################
            EVS = explained_variance_score(y_test, mean_prediction)
            MAE = mean_absolute_error(y_test, mean_prediction)
            MSE = mean_squared_error(y_test, mean_prediction)
            RMSE_KPA = (math.sqrt(mean_squared_error(y_test, mean_prediction))/(pow(0.004,2)*math.pi))/pow(10,3)
            R2 = r2_score(y_test, mean_prediction)
            ############################################################################
            # save the model
            df_train = df_train.append({"case_num": str(case), "RH": str(RH_num), "folder_name": str(sensor_name), "sample_num": NUM_MIN_SAMPLE,"time": float(end_time - start_time), "pred_time":float(pred_end_time - pred_start_time) , "explained_variance_score": format(float(EVS), '.5f'), "mean_absolute_error": format(float(MAE), '.5f'), "mean_squared_error": format(float(MSE), '.5f'), "RMSE_KPA": format(float(RMSE_KPA), '.5f'), "R2": format(float(R2), '.5f')}, ignore_index=True)
    #         ##############################################################
    #         ##############################################################
    #         ########        Test data                      ###############
    #         ############################################################## 
    #         ##############################################################
    #          # test data path
            
             
    #         raspi_path = RH_name_path+"RasPi/"
    #         ##########################################################
    #         ##########################################################
    #         # stance data
            
    #         if any(RH_num == RH_number for RH_number in ["02","04","06","07"]):
    #             pass
    #         else:
            
    #             stance_path = str(raspi_path)+"stance_data/*"
    #             stance_name_path = str(raspi_path)+"stance_data/"
                
    #             stance_data_list = glob.glob(stance_path)
    #             stance_data_list = [file.replace('\\','/') for file in stance_data_list]
    #             stance_data_name_list = os.listdir(stance_name_path)
                
    #             stance_data_list_ = [file for (file, name) in zip(stance_data_list, stance_data_name_list)  if (name.startswith("stance")==1)]
    #             stance_data_name_list_ = [name for (file, name) in zip(stance_data_list, stance_data_name_list)  if (name.startswith("stance") == 1)]
                
    #             for (stance_file, stance_name) in zip(stance_data_list_, stance_data_name_list_):
                    
    #                 stance_num = stance_name[7:]
                    
    #                 ######################################################
    #                 # LorR data
    #                 stance_final_path = str(stance_file)+"/%s/*" %(str(sensor_dir[0].upper()))
    #                 stance_final_name_path = str(stance_file)+"/%s/" %(str(sensor_dir[0].upper()))
                    
    #                 stance_final_list = glob.glob(stance_final_path)
    #                 stance_final_list = [file.replace('\\','/') for file in stance_final_list]
    #                 stance_final_name_list = os.listdir(stance_final_name_path)
                    
    #                 for (stance_final_file, stance_final_name) in zip(stance_final_list, stance_final_name_list):
                        
    #                     if (stance_final_name.startswith("N") == 1) & (stance_final_name.endswith(str(sensor_num)+".csv") == 1):
                            
    #                         test_data_path = stance_final_file
    #                         raw_test_data = pd.read_csv(test_data_path, header = 0)
    #                         test_data = raw_test_data.astype(float)
                            
    #                         # 결측치 제거
    #                         reg_test_data = test_data.dropna(axis=0)
    #                         reg_test_data.reset_index(drop=True, inplace=True)
                            
    #                         # class별 minimum number of samples
    #                         NUM_MIN_SAMPLE = int(len(reg_test_data))
                            
    #                         # scaler 적용
    #                         # reg_data[["vout", "force"]] = StandardScaler().fit_transform(reg_data[["vout", "force"]])
                            
    #                         # case 1
    #                         # X = reg_data[["vout", "load_unload", "cycle_num", "disp_rate", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "indenter_pos"]]
    #                         # case 2
    #                         # X = reg_data[["vout", "load_unload", "cycle_num", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                         # X = reg_data[["vout", "loading_type", "cycle_num", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                         # case 3
    #                         # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                         X = reg_test_data[X_list]
    #                         # case 4
    #                         # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1", "pre_2", "pre_3"]]
    #                         # case 5
    #                         # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1"]]
    #                         # case 6
    #                         # X = reg_data[["vout", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                         # case 7
    #                         # X = reg_test_data[["vout"]].values.reshape(-1,1)
    #                         ###############################################################
    #                         # N-dimensional GPR
    #                         ##############################################################
    #                         pred_start_time = time()
                            
    #                         mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
                            
    #                         pred_end_time = time()
    #                         # save the model
    #                         df_test = df_test.append({"case_num": str(case),"RH": str(RH_num), "stance_or_walk": "stance", "stance_or_walk_num": str(stance_num), "sensor_dir": str(sensor_dir[0].upper()),"sensor_num": str(sensor_num),"sample_num": NUM_MIN_SAMPLE, "pred_time":float(pred_end_time - pred_start_time)}, ignore_index=True)
                            
    #                         pred_directory = str(stance_final_name_path)+"prediction"
                            
    #                         try:
    #                             if not os.path.exists(pred_directory):
    #                                 os.makedirs(pred_directory)
    #                         except OSError:
    #                             pass
                            
    #                         final_test_data = pd.DataFrame(data = reg_test_data, columns = reg_test_data.columns)
    #                         final_test_data["mean_prediction"] = mean_prediction
    #                         final_test_data["std_prediction"] = std_prediction
    #                         final_test_data.to_csv(pred_directory+"/pred_SENSOR_%s.csv" %(str(sensor_num)),sep=',', index = False, header = True)
    #                 pass         
    #         ##########################################################
    #         ##########################################################
    #         # walk data
    #         walk_path = str(raspi_path)+"walk_data/*"
    #         walk_name_path = str(raspi_path)+"walk_data/"
            
    #         walk_data_list = glob.glob(walk_path)
    #         walk_data_list = [file.replace('\\','/') for file in walk_data_list]
    #         walk_data_name_list = os.listdir(walk_name_path)
            
    #         walk_data_list_ = [file for (file, name) in zip(walk_data_list, walk_data_name_list)  if (name.startswith("walk")==1)]
    #         walk_data_name_list_ = [name for (file, name) in zip(walk_data_list, walk_data_name_list)  if (name.startswith("walk") == 1)]
            
    #         for (walk_file, walk_name) in zip(walk_data_list_, walk_data_name_list_):
                
    #             walk_num = walk_name[5:]
                
    #             ######################################################
    #             # LorR data
    #             walk_final_path = str(walk_file)+"/%s/*" %(str(sensor_dir[0].upper()))
    #             walk_final_name_path = str(walk_file)+"/%s/" %(str(sensor_dir[0].upper()))
                
    #             walk_final_list = glob.glob(walk_final_path)
    #             walk_final_list = [file.replace('\\','/') for file in walk_final_list]
    #             walk_final_name_list = os.listdir(walk_final_name_path)
                
    #             for (walk_final_file, walk_final_name) in zip(walk_final_list, walk_final_name_list):
                    
    #                 if (walk_final_name.startswith("N") == 1) & (walk_final_name.endswith(str(sensor_num)+".csv") == 1):
                        
    #                     test_data_path = walk_final_file
    #                     raw_test_data = pd.read_csv(test_data_path, header = 0)
    #                     test_data = raw_test_data.astype(float)
                        
    #                     # 결측치 제거
    #                     reg_test_data = test_data.dropna(axis=0)
    #                     reg_test_data.reset_index(drop=True, inplace=True)
                        
    #                     # class별 minimum number of samples
    #                     NUM_MIN_SAMPLE = int(len(reg_test_data))
                        
    #                     # scaler 적용
    #                     # reg_data[["vout", "force"]] = StandardScaler().fit_transform(reg_data[["vout", "force"]])
                        
    #                     # case 1
    #                     # X = reg_data[["vout", "load_unload", "cycle_num", "disp_rate", "del_V", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5", "indenter_pos"]]
    #                     # case 2
    #                     # X = reg_data[["vout", "load_unload", "cycle_num", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                     # X = reg_data[["vout", "loading_type", "cycle_num", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                     # case 3
    #                     # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                     X = reg_test_data[X_list]
    #                     # case 4
    #                     # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1", "pre_2", "pre_3"]]
    #                     # case 5
    #                     # X = reg_data[["vout", "load_unload", "disp_rate", "pre_1"]]
    #                     # case 6
    #                     # X = reg_data[["vout", "pre_1", "pre_2", "pre_3", "pre_4", "pre_5"]]
    #                     # case 7
    #                     # X = reg_test_data[["vout"]].values.reshape(-1,1)
    #                     ###############################################################
    #                     # N-dimensional GPR
    #                     ##############################################################
    #                     pred_start_time = time()
                        
    #                     mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
                        
    #                     pred_end_time = time()
    #                     # save the model
    #                     df_test = df_test.append({"case_num": str(case),"RH": str(RH_num), "stance_or_walk": "walk", "stance_or_walk_num": str(walk_num), "sensor_dir": str(sensor_dir[0].upper()),"sensor_num": str(sensor_num),"sample_num": NUM_MIN_SAMPLE, "pred_time":float(pred_end_time - pred_start_time)}, ignore_index=True)
                        
    #                     pred_directory = str(walk_final_name_path)+"prediction"
                        
    #                     try:
    #                         if not os.path.exists(pred_directory):
    #                             os.makedirs(pred_directory)
    #                     except OSError:
    #                         pass
                        
    #                     final_test_data = pd.DataFrame(data = reg_test_data, columns = reg_test_data.columns)
    #                     final_test_data["mean_prediction"] = mean_prediction
    #                     final_test_data["std_prediction"] = std_prediction
    #                     final_test_data.to_csv(pred_directory+"/pred_SENSOR_%s.csv" %(str(sensor_num)),sep=',', index = False, header = True)
                    
df_train.to_csv(base_path + "df_train_N_total.csv", sep=",", header=True, index=False)
    # df_test.to_csv(RH_name_path+"df_test_N_total.csv",sep=",",header=True, index=False)