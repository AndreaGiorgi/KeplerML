#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
"""
import itertools
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt  

# TODO
# 1. Implement ETL pipeline
# 2. Implement Feature selection algorithm
# 3. Implement SVM model with Sigmoid Kernel
# 4. Results visualization 

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

def load_training_planets_data():
    
    habitable_planets = np.genfromtxt('../data/habitable_planets_detailed_list.csv',filling_values = 0, names=True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes, encoding=None) 
    non_habitable_planets = np.genfromtxt('../data/non_habitable_planets_confirmed_detailed_list.csv', filling_values = 0, names = True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes, encoding=None) 
    
    np.random.shuffle(habitable_planets)
    np.random.shuffle(non_habitable_planets)        

    return habitable_planets, non_habitable_planets

def get_model(X,y):
    
    temp_model = svm.SVC(kernel='sigmoid')
    
    sfs = SequentialFeatureSelector(temp_model, n_features_to_select=12)
    sfs.fit(X,y)
    X_new = sfs.transform(X)
    optimized_model = svm.SVC(kernel='sigmoid', class_weight='balanced', random_state=42)
    
    optimized_model.fit(X,y)
    return optimized_model

def select_features(from_data, to_data):
    for i in planetary_stellar_parameter_cols:
        to_data = np.column_stack((to_data, from_data[i])) 
        
    return to_data  


def get_trained_model():

    habitable, non_habitable = load_training_planets_data()

    habitable_slice_features = np.ones(habitable.shape[0])     
    non_habitable_slice_features = np.full(non_habitable.shape[0], -1) 
     
    habitable_slice_features = select_features(habitable, habitable_slice_features) 
    non_habitable_slice_features = select_features(non_habitable, non_habitable_slice_features) 
     
    X_train = np.vstack((habitable_slice_features[:,1:], non_habitable_slice_features[:,1:]))  
    Y_train = np.append(habitable_slice_features[:,0], non_habitable_slice_features[:,0])

    model = get_model(X_train, Y_train)
    
    return model

def predict_on_new_kepler_data(kepler_data_file):
    
    svm_model = get_trained_model()
    kepler_planets = np.genfromtxt(kepler_data_file, filling_values = 0, names=True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes, encoding=None)

    X_data = np.ndarray(shape=(kepler_planets.shape[0],0)) 
    X_data = select_features(kepler_planets, X_data)

    y_predicted = svm_model.predict(X_data) 
    
    X_distance_from_parent_star = []
    Y_surface_temprature = []
    S_planet_radius = []
    colors = []
    color_space = 255*255*255
    
    total_distance = 0
    total_temperature = 0
    number_of_habitable_planets = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] > 0:
            habitable_planet_koi = kepler_planets[i]["kepoi_name"]
            planet_temperature = kepler_planets[i]["koi_teq"] - 273.15
            total_temperature += planet_temperature
            planet_radius = kepler_planets[i]["koi_prad"] 
            planet_star_distance = kepler_planets[i]["koi_dor"]
            total_distance += planet_star_distance
            number_of_habitable_planets += 1
            print('Predicted Habitable planet koi = ',habitable_planet_koi, ", Equilibrium Temperature in Celsius = ", planet_temperature, ", Planet radius (Earth) = ", planet_radius)         
            X_distance_from_parent_star.append(planet_star_distance)
            Y_surface_temprature.append(planet_temperature)
            S_planet_radius.append(planet_radius)
            colors.append(np.random.randint(color_space))
    
    print("features used were ", planetary_stellar_parameter_cols)
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

def main():
    parser = argparse.ArgumentParser(description='Predict habitability on kepler cumulative data, or test model on training data.')
    parser.add_argument('--predict_kepler_file', nargs=1, help=' please pass location of kepler cumulative data file. If this argument is not passed a simple training on train data will occur based on kernel type choosen ') 
    
    current_args = parser.parse_args()
    #kepler_data_file = current_args.predict_kepler_file
    kepler_data_file = '../data/cumulative_test.csv'

    print('This might take few minutes to run')
    predict_on_new_kepler_data(kepler_data_file)

main()      

