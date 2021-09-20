#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ramisra
"""

import itertools
import argparse
import os
import time
import numpy as np
import psutil
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix  
import matplotlib.pyplot as plt       

TRAIN_DATA = 0.5 
DEV_DATA = 0.2 
BEST_FEATURE_SELECTION_LOOP_COUNT= 10

#Index is 0 based
# Planetary and Stellar parameters    
planetary_stellar_parameter_indexes = (2,   # kepoi_name:      KOI Name
                                       15,  # koi period,      Orbital Period [days]
                                       42,  # koi_ror:         Planet-Star Radius Ratio
                                       45,  # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                       49,  # koi_prad:        Planetary Radius [Earth radii]
                                       52,  # koi_sma:         Orbit Semi-Major Axis [AU]
                                       58,  # koi_teq:         Equilibrium Temperature [K]
                                       61,  # koi_insol:       Insolation Flux [Earth flux]
                                       64,  # koi_dor:         Planet-Star Distance over Star Radius
                                       76,  # koi_count:       Number of Planet 
                                       87,  # koi_steff:       Stellar Effective Temperature [K] 
                                       90,  # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                       93,  # koi_smet:        Stellar Metallicity [dex]
                                       96,  # koi_srad:        Stellar Radius [Solar radii]
                                       99   # koi_smass:       Stellar Mass [Solar mass]
                                       )
#Names of columns from kepler data
planetary_stellar_parameter_cols = (   "koi_period",    # koi_period       Orbital Period [days]
                                       "koi_ror",       # koi_ror:         Planet-Star Radius Ratio
                                       "koi_srho",      # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                       "koi_prad",      # koi_prad:        Planetary Radius [Earth radii]
                                       "koi_sma",       # koi_sma:         Orbit Semi-Major Axis [AU]
                                       "koi_teq",       # koi_teq:         Equilibrium Temperature [K]
                                       "koi_insol",     # koi_insol:       Insolation Flux [Earth flux]
                                       "koi_dor",       # koi_dor:         Planet-Star Distance over Star Radius
                                       "koi_count",     # koi_count:       Number of Planet 
                                       "koi_steff",     # koi_steff:       Stellar Effective Temperature [K] 
                                       "koi_slogg",     # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                       "koi_smet",      # koi_smet:        Stellar Metallicity [dex]
                                       "koi_srad",      # koi_srad:        Stellar Radius [Solar radii]
                                       "koi_smass"      # koi_smass:       Stellar Mass [Solar mass]
                                       )

planetary_stellar_parameter_cols_dict = {   "koi_period":   "Orbital Period",
                                       "koi_ror":     "Planet-Star Radius Ratio",
                                       "koi_srho":      "Fitted Stellar Density",
                                       "koi_prad":     "Planetary Radius",
                                       "koi_sma":      "Orbit Semi-Major Axis",
                                       "koi_teq":       "Equilibrium Temperature",
                                       "koi_insol":     "Insolation Flux",
                                       "koi_dor":       "Planet-Star Distance over Star Radius",
                                       "koi_count":     "Number of Planet" ,
                                       "koi_steff":     "Stellar Effective Temperature" ,
                                       "koi_slogg":     "Stellar Surface Gravity",
                                       "koi_smet":      "Stellar Metallicity",
                                       "koi_srad":      "Stellar Radius",
                                       "koi_smass":      "Stellar Mass"
                                       }
'''
This method takes training X,y and removes
unnecessary features using Recursive feature
reduction with cross validation and returns
trained SVM with linear kernel
'''
def get_linear_svm_kernel_with_feature_reduction(X, y):
    print('Running recursive feature elimination with cross validation with linear SVM')
   
    svc = svm.SVC(C = 1, kernel='linear', class_weight = 'balanced')
    svc.fit(X,y)
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)
    return rfecv

'''
This method test how linear kernel
with feature reduction performs on
unseen test data.
'''
def test_feature_reduction():
    print('Training model on test data with feature elimination and validating it on test data')
    habitable_planets , non_habitable_planets = load_training_planets_data()
    X_train, Y_train = get_X_Y(habitable_planets, non_habitable_planets, planetary_stellar_parameter_cols, 0, 0.7)
    
    X_test, Y_test = get_X_Y(habitable_planets, non_habitable_planets, planetary_stellar_parameter_cols, 0.7, 0.3)
    
    clf = get_linear_svm_kernel_with_feature_reduction(X_train, Y_train)
    
    Y_predict = clf.predict(X_test)
    draw_confusion_matrix(Y_predict, Y_test, "feature elimination")    

    result = Y_predict * Y_test
    error = (sum(1 for i in result if i <= 0)/len(Y_test))*100
    
    print('Test Error ' , error)    

    Y_predict = clf.predict(X_train)

    result = Y_predict * Y_train 
    error = (sum(1 for i in result if i <= 0)/len(Y_train))*100 
    
    print('Training Error ' , error)    

def select_features(from_data, to_data, feature_indexes):
    for i in feature_indexes:
        to_data = np.column_stack((to_data, from_data[i])) 
        
    return to_data     

'''
Given planet's data and indexes of features, it 
extracts features from data with all features,
and adds y = (-1, 1) and returns X,y
'''                                    
def get_X_Y(habitable, non_habitable, feature_indexes, start, width):
    
    habitable_size = len(habitable) 
    non_habitable_size = len(non_habitable) 
    
    habitable_slice = habitable[int(habitable_size*start):int(habitable_size*(start+width))]     
    non_habitable_slice = non_habitable[int(non_habitable_size*start):int(non_habitable_size*(start+width))] 
    
    habitable_slice_features = np.ones(len(habitable_slice))     
    non_habitable_slice_features = np.full((len(non_habitable_slice)), -1) 
    
    habitable_slice_features = select_features(habitable_slice, habitable_slice_features, feature_indexes) 
    non_habitable_slice_features = select_features(non_habitable_slice, non_habitable_slice_features, feature_indexes) 
    
    X = np.vstack((habitable_slice_features[:,1:], non_habitable_slice_features[:,1:]))  
    Y = np.append(habitable_slice_features[:,0], non_habitable_slice_features[:,0]) 
    
    return X, Y 

def do_svm(X_train, Y_train, X_predict):
    
    clf = svm.SVC(C = 1, kernel='rbf', gamma = 10, class_weight = 'balanced')
    clf.fit(X_train, Y_train)
        
    y_predicted = clf.predict(X_predict) 
    return y_predicted 

'''
Draws a simple confusion matrix with X axis
predicted habitable and non habitable planets
and Y axis with true labels.
'''
def draw_confusion_matrix(y_predicted, Y_actual, title):
    cm = confusion_matrix(Y_actual, y_predicted)
    
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.ocean)
    plt.title("Confusion matrix " + title)
    plt.colorbar()
    classes = ("non habitable", "habitable")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
'''
This is wrapper method to do train data and do test
on given test data.  Test data could be dev data or
actual test data.
'''    
def get_test_error(habitable_planets, non_habitable_planets, features, start_train, train_width, start_test, test_width):
    X_train, Y_train = get_X_Y(habitable_planets, non_habitable_planets, features, start_train, train_width) 
    X_dev, Y_dev = get_X_Y(habitable_planets, non_habitable_planets, features, start_test, test_width) 
                
    y_predicted = do_svm(X_train, Y_train, X_dev)         
    result = y_predicted * Y_dev 
    
    error = (sum(1 for i in result if i <= 0)/len(Y_dev))*100 
    
    return error, y_predicted, Y_dev     

'''
This method implements forward search of features.  It adds 
one feature at a time and selects feature with lowest
dev error
'''    
def forward_search_features (habitable, non_habitable, start_train, train_width, start_test, test_width):
    selected_features = set([]) 
    previous_min_error = 0 
    for i in planetary_stellar_parameter_cols:
        min_error = 100         
        min_index = 0 
        for j in planetary_stellar_parameter_cols:
            if j not in selected_features:
                tmp_selected_features = set(selected_features) 
                tmp_selected_features.add(j) 
                
                error,_,_ = get_test_error(habitable, non_habitable, tmp_selected_features, start_train, train_width, start_test, test_width) 
                
                if error < min_error:
                    min_index = j 
                    min_error = error 
        
        if previous_min_error == min_error:
            break 
            
        selected_features.add(min_index) 
        previous_min_error = min_error 
    
    return selected_features 

def load_training_planets_data():
    habitable_planets = np.genfromtxt('../data/habitable_planets_detailed_list.csv',filling_values = 0, names=True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes, encoding=None) 
    non_habitable_planets = np.genfromtxt('../data/non_habitable_planets_confirmed_detailed_list.csv', filling_values = 0, names = True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes, encoding=None) 
    
    np.random.shuffle(habitable_planets)
    np.random.shuffle(non_habitable_planets)        

    return habitable_planets, non_habitable_planets 

'''
This tries to find the features which represent lowest
overall dev error. Distribution of data sometimes
determine which features are selected. To avoid
relying on result of one particular data, multiple
iteration of logic is performed, each time
reshuffling train and dev data (test data is never 
touched) and at the end find feature which lowest
dev error.  This is probably not the perfect logic
as it might introduce a feature set overfitting to 
one set of dev data that gave lowest error and
may not represent larger section of data.
'''
def find_best_features():
    try:
        print("Selecting best features") 
    
        dictionary_of_features = dict() 
        habitable_planets , non_habitable_planets = load_training_planets_data()  
        
        for j in range(BEST_FEATURE_SELECTION_LOOP_COUNT):
            habitable_size = len(habitable_planets) 
            non_habitable_size = len(non_habitable_planets) 

            #only resuffule train/dev set  
            np.random.shuffle(habitable_planets[0: int(habitable_size * (TRAIN_DATA + DEV_DATA))])
            np.random.shuffle(non_habitable_planets[0:int(non_habitable_size * (TRAIN_DATA + DEV_DATA))])        
            
            selected_features = forward_search_features(habitable_planets, non_habitable_planets, 0.0, TRAIN_DATA, TRAIN_DATA, DEV_DATA) 
            frozen_selected_features = frozenset(selected_features) 
            if frozen_selected_features not in dictionary_of_features:
               dictionary_of_features[frozen_selected_features] = 1 
            else:
               dictionary_of_features[frozen_selected_features] = dictionary_of_features[frozen_selected_features] + 1 
            
            print('.', end='', flush=True) 

        # select top 4 set
        TOP_NUMBER_OF_FEATURES = 4
        index = 0 
        min_dev_error = 100 
        best_feature_set = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for key, value in sorted(dictionary_of_features.items(), key=lambda x:x[1], reverse=True):
            if index == TOP_NUMBER_OF_FEATURES:
                break 
            index +=1
            
            dev_error,_,_ = get_test_error(habitable_planets, non_habitable_planets, key, 0.0, TRAIN_DATA, TRAIN_DATA, DEV_DATA)
            feature_label = []
            for feature in key:
                feature_label.append(planetary_stellar_parameter_cols_dict[feature])
            ax.scatter(dev_error, value, label=feature_label)
            if dev_error < min_dev_error:
                 best_feature_set = key
                 min_dev_error = dev_error
        
        Y_annotate = dictionary_of_features[best_feature_set]
        X_annotate = min_dev_error
        
        ax.set_xlim([0, 20])
        ax.set_ylim([0, BEST_FEATURE_SELECTION_LOOP_COUNT])
        best_feature_label = "{"
        for feature in best_feature_set:
                best_feature_label += " {" + planetary_stellar_parameter_cols_dict[feature] + "} "
        best_feature_label += "}"
                
        ax.annotate('Feature selected = ' + best_feature_label, xy = (X_annotate, Y_annotate), xytext = (X_annotate + 2, Y_annotate + 2), 
                     arrowprops=dict(facecolor='black', shrink=0.05),)
           
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           mode="expand", borderaxespad=0.)
        plt.xlabel('% Dev set error with selected feature')
        plt.ylabel('Number of times feature selected')
        plt.show()
       
        print("\nBest selected features are " , best_feature_set)
        return best_feature_set, habitable_planets, non_habitable_planets 
    
    except ValueError:
        print('Error reading file') 
        raise 

def test_features():                                       
    try:
        best_features, habitable_planets, non_habitable_planets = find_best_features() 
        
        #error on test data
        test_error, y_predicted, y_actual = get_test_error(habitable_planets, non_habitable_planets, best_features, 0.0, TRAIN_DATA, TRAIN_DATA + DEV_DATA, 1.0) 
        draw_confusion_matrix(y_predicted, y_actual , "for test data")
        print('\ntest error on test data is ', test_error) 
        
        # error on train data
        train_error, y_predicted, y_actual = get_test_error(habitable_planets, non_habitable_planets, best_features, 0.0, TRAIN_DATA, 0.0, TRAIN_DATA) 
        print('Error on training data is ', train_error) 

        # error on dev data
        dev_error, y_predicted, y_actual = get_test_error(habitable_planets, non_habitable_planets, best_features,  0.0, TRAIN_DATA, TRAIN_DATA, DEV_DATA) 
        print('Error on dev data is ', dev_error) 
         
    except ValueError:
        print('Error reading file') 
        raise 
        
def get_trained_model(kernel):
     if kernel == 'rbf':
         best_features, habitable_planets, non_habitable_planets  = find_best_features() 
     else:
         habitable_planets , non_habitable_planets = load_training_planets_data() 
         best_features = planetary_stellar_parameter_cols

     habitable_slice_features = np.ones(habitable_planets.shape[0])     
     non_habitable_slice_features = np.full(non_habitable_planets.shape[0], -1) 
     
     habitable_slice_features = select_features(habitable_planets, habitable_slice_features, best_features) 
     non_habitable_slice_features = select_features(non_habitable_planets, non_habitable_slice_features, best_features) 
     
     X_train = np.vstack((habitable_slice_features[:,1:], non_habitable_slice_features[:,1:]))  
     Y_train = np.append(habitable_slice_features[:,0], non_habitable_slice_features[:,0]) 
     
     if kernel == 'rbf':
        param_grid = {'C': [1, 5, 10], 'gamma': [10, 'auto', 'scale'],'kernel': ['rbf'], 'class_weight': ['balanced']}
        clf = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
        clf.fit(X_train,Y_train)
        print(clf.best_params_)
     else:
         clf = get_linear_svm_kernel_with_feature_reduction(X_train, Y_train) 
     
     return clf, best_features 

'''
Method used to train model and later on run that
trained model on data file generated from 
Kepler mission's exoplanet archieve
''' 
def predict_on_new_kepler_data(kepler_data_file, kernel):
    clf, features = get_trained_model(kernel)

    planets_from_kepler = np.genfromtxt(kepler_data_file, filling_values = 0, names=True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes, encoding=None) 
    
    X_data = np.ndarray(shape=(planets_from_kepler.shape[0],0)) 

    X_data = select_features(planets_from_kepler, X_data, features) 
    
    y_predicated = clf.predict(X_data) 
    
    X_distance_from_parent_star = []
    Y_surface_temprature = []
    S_planet_radius = []
    colors = []
    color_space = 255*255*255
    
    total_distance = 0
    total_temperature = 0
    number_of_habitable_planets = 0
    for i in range(len(y_predicated)):
        if y_predicated[i] > 0:
            habitable_planet_koi = planets_from_kepler[i]["kepoi_name"]
            planet_temperature = planets_from_kepler[i]["koi_teq"] - 273.15
            total_temperature += planet_temperature
            planet_radius = planets_from_kepler[i]["koi_prad"] 
            planet_star_distance = planets_from_kepler[i]["koi_dor"]
            total_distance += planet_star_distance
            number_of_habitable_planets += 1
            print('Predicted Habitable planet koi = ',habitable_planet_koi, ", Equilibrium Temperature in Celsius = ", planet_temperature, ", Planet radius (Earth) = ", planet_radius)         
            X_distance_from_parent_star.append(planet_star_distance)
            Y_surface_temprature.append(planet_temperature)
            S_planet_radius.append(planet_radius)
            colors.append(np.random.randint(color_space))
    
    print("features used were ", features)
    print("Number of habitable planets detected " , number_of_habitable_planets)
    mean_distance = total_distance/number_of_habitable_planets
    print('Mean distance of habitable planets ' , mean_distance)
    var_distance = np.sqrt(np.sum(np.square(X_distance_from_parent_star - mean_distance))/number_of_habitable_planets)
    print('Standard deviation of distance of habitable planets ' , var_distance)
    
    mean_temp = total_temperature/number_of_habitable_planets
    print('Mean temperature of habitable planets ' , mean_temp)
    var_temp = np.sqrt(np.sum(np.square(Y_surface_temprature - mean_temp))/number_of_habitable_planets)
    print('Standard deviation temperature of habitable planets ' , var_temp)
    
    plt.scatter(X_distance_from_parent_star, Y_surface_temprature, s = S_planet_radius, c = colors)
    plt.xlabel('Distance from parent star in the unit of star\'s radius')
    plt.ylabel('Planetary Equilibrium Temperature in Celsius')
    plt.show()
    
    X_distance_from_parent_star_verified = []
    Y_surface_temprature_verified = []
    S_planet_radius_verified = []
    colors_verified = []
    
    minimum_temperature = 0
    maximum_temperature = 0    
    number_of_confirmed_habitable_planets = 0
    
    for i in range(len(y_predicated)):
        if y_predicated[i] == 1:
            habitable_planet_koi = planets_from_kepler.iloc[i, 2] #kepoi_name: Planet Kepler code
            planet_temperature = planets_from_kepler.iloc[i, 58] #koi_teq: Planet temperature converted from Kelvin to Celsius
            planet_temperature_celsius = planets_from_kepler.iloc[i, 58] - 273.15  #koi_teq in Celsius 
            planet_radius = planets_from_kepler.iloc[i, 49] #koi_prad: Planet Radius
            planet_star_distance = planets_from_kepler.iloc[i, 64] #koi_dor: Distance from Planet to Star measured in Earth-Sun distance
    
            # This filter will plot only predicted habitable planets which main features are between habitability range. We have followed 
            # NASA and ESA habitability definitions in order to show only "candidate" planets which are inside the cumulative dataset. 
            # The cumulative dataset, which is used as test set, contains all exoplanets found by Kepler. 
            # #
            
    if 183 <= planet_star_distance <= 460 and 200 <= planet_temperature <= 600 and 0.5 <= planet_radius <= 3.3:
        print('Predicted "confirmed" Habitable planet koi = ',habitable_planet_koi, ", Equilibrium Temperature in Celsius = ", planet_temperature_celsius, ", Planet radius (Earth) = ", planet_radius)         
        X_distance_from_parent_star_verified.append(planet_star_distance)
        Y_surface_temprature_verified.append(planet_temperature_celsius)
        S_planet_radius_verified.append(planet_radius)
        colors_verified.append(np.random.randint(color_space))
        number_of_confirmed_habitable_planets += 1

    print('Predicted habitable planets inside habitable ranges: ', number_of_confirmed_habitable_planets)
    plt.scatter(X_distance_from_parent_star_verified, Y_surface_temprature_verified, s = S_planet_radius_verified, c = colors_verified)
    plt.xlabel('Distance from parent star, in Earth-Sun distance')
    plt.ylabel('Planetary Equilibrium Temperature in Celsius')
    plt.show()  

'''
This program could be called either with just --kernel = {linear, rbf}
or data file with all columns generated from Kepler's KOI table
 
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

The first line of data file needs to be column names.

One example of how to run this is

python predict_habitability.py --predict_kepler_file  ../data/cumulative_test.csv --kernel linear

python predict_habitability.py --predict_kepler_file  ../data/cumulative_test.csv --kernel rbf

In this case, it prints KOI of all planets which it has
identified as potentially habitable and scatter plot
of habitable planets identified.

If this is called without --predict_kepler_file argument, it just finds best feature from
training/dev data and reports the error on training and test data.

python predict_habitability.py --kernel linear
python predict_habitability.py --kernel rbf

'''

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()/1024/1024
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        mem_after = get_process_memory()/1024/1024
        print("{}: memory before: {:,} MB, after: {:,} MB, consumed: {:,} MB; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, (mem_after - mem_before),
            elapsed_time))
        return result
    return wrapper

@track
def main():
    parser = argparse.ArgumentParser(description='Predict habitability on kepler cumulative data, or test model on training data.')
    parser.add_argument('--predict_kepler_file', nargs=1, help=' please pass location of kepler cumulative data file. If this argument is not passed a simple training on train data will occur based on kernel type choosen ')
    parser.add_argument('--kernel', nargs='?',choices=['linear', 'rbf'], required=True, help='Pass kernel as either linear or rbf ')
        
    current_args = parser.parse_args()
    kernel =  current_args.kernel
    kepler_data_file = current_args.predict_kepler_file
    print('This might take few minutes to run')
    if kepler_data_file is not None:
            predict_on_new_kepler_data(kepler_data_file[0], kernel)
    else:
        if kernel == 'linear':
            test_feature_reduction()
        else:
            test_features()

main()      