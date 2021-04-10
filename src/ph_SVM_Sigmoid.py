#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Andrea Giorgi and Gianluca De Angelis
"""

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

planetary_stellar_parameter_indexes = (2,  # kepoi_name:      KOI Name
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
                                       99  # koi_smass:       Stellar Mass [Solar mass]
                                       )
# Names of columns from kepler data
planetary_stellar_parameter_cols = ("koi_period",  # koi_period       Orbital Period [days]
                                    "koi_ror",  # koi_ror:         Planet-Star Radius Ratio
                                    "koi_srho",  # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                    "koi_prad",  # koi_prad:        Planetary Radius [Earth radii]
                                    "koi_sma",  # koi_sma:         Orbit Semi-Major Axis [AU]
                                    "koi_teq",  # koi_teq:         Equilibrium Temperature [K]
                                    "koi_insol",  # koi_insol:       Insolation Flux [Earth flux]
                                    "koi_dor",  # koi_dor:         Planet-Star Distance over Star Radius
                                    "koi_count",  # koi_count:       Number of Planet
                                    "koi_steff",  # koi_steff:       Stellar Effective Temperature [K]
                                    "koi_slogg",  # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                    "koi_smet",  # koi_smet:        Stellar Metallicity [dex]
                                    "koi_srad",  # koi_srad:        Stellar Radius [Solar radii]
                                    "koi_smass"  # koi_smass:       Stellar Mass [Solar mass]
                                    )

planetary_stellar_parameter_cols_dict = {"koi_period": "Orbital Period",
                                         "koi_ror": "Planet-Star Radius Ratio",
                                         "koi_srho": "Fitted Stellar Density",
                                         "koi_prad": "Planetary Radius",
                                         "koi_sma": "Orbit Semi-Major Axis",
                                         "koi_teq": "Equilibrium Temperature",
                                         "koi_insol": "Insolation Flux",
                                         "koi_dor": "Planet-Star Distance over Star Radius",
                                         "koi_count": "Number of Planet",
                                         "koi_steff": "Stellar Effective Temperature",
                                         "koi_slogg": "Stellar Surface Gravity",
                                         "koi_smet": "Stellar Metallicity",
                                         "koi_srad": "Stellar Radius",
                                         "koi_smass": "Stellar Mass"
                                         }


def data_visualization_analysis(planets, features, clf, X_test, predictions, test_values):
    
    X_distance_from_parent_star = []
    Y_surface_temprature = []
    S_planet_radius = []
    colors = []
    color_space = 255*255*255
    
    planets = planets.fillna(0)
    
    total_distance = 0
    total_temperature = 0
    number_of_habitable_planets = 0
    for i in range(len(predictions)):
        if predictions[i] > 0:
            habitable_planet_koi = planets.iloc[i, 2] #kepoi_name
            planet_temperature = planets.iloc[i, 58]  #koi_teq
            total_temperature += planet_temperature - 273.15
            planet_radius = planets.iloc[i, 49] #koi_prad
            planet_star_distance = planets.iloc[i, 64] #koi_dor
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
    
    print('Accuracy Score : ' + str(accuracy_score(test_values, predictions)))
    print('Precision Score : ' + str(precision_score(test_values, predictions)))
    print('Recall Score : ' + str(recall_score(test_values, predictions)))
    print('F1 Score : ' + str(f1_score(test_values, predictions)))
    plot_confusion_matrix(clf, X_test, test_values, cmap=plt.cm.YlGn)
    plt.show()


def dataset_normalization(x_train, x_test, method):
    
    if method == 'standard':
        scaler = StandardScaler()
        normalized_train = scaler.fit_transform(x_train)
        normalized_test = scaler.fit_transform(x_test)
    else: 
        scaler = MinMaxScaler()
        normalized_train = scaler.fit_transform(x_train)
        normalized_test = scaler.fit_transform(x_test)
    return normalized_train, normalized_test

def dataset_encoding(data):

    X = data
    features_to_encode = X.dtypes == object
    columns = X.columns[features_to_encode].tolist()
    le = LabelEncoder()
    X[columns] = X[columns].apply(lambda col: le.fit_transform(col))

    return X


def dataset_processing(dataset):
    
    ## Drop all columns that are not listed in Planetary columns
    
    planetary_stellar_features = ["Habitable", "koi_period", "koi_ror", "koi_srho", "koi_prad", 
                                  "koi_sma", "koi_teq", "koi_insol", "koi_dor", "koi_count", 
                                  "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass"]
    dataset = dataset[planetary_stellar_features]
    
    ## Visualize and fill all missing and NaN data
    
    missing_data = dataset.isnull()
    for column in dataset:
        print(column)
        print(missing_data[column].value_counts())
        print('')
        
    NaN_data = dataset.isna()
    for column in dataset:
        print(column)
        print(NaN_data[column].value_counts())
        print('')
        
    dataset = dataset.fillna(0)

    ## Now we can analyze correlation between Habitable and remaining features (USELESS)
    correlation = dataset.corr()
    sns.heatmap(correlation, 
            xticklabels=correlation.columns.values,
            yticklabels=correlation.columns.values)
    
    selected = ['koi_prad','koi_insol']   #RBF feature selection on cumulative_test (C=5, balanced, gamma 10)
    
    encoded_dataset = dataset_encoding(dataset)
    
    return encoded_dataset, selected


def dataset_loading():
    dataset = pd.read_csv('data/cumulative_test.csv')
    habitable_planets = pd.read_csv('data/habitable_planets_detailed_list.csv')
    dataset = pd.concat([dataset, habitable_planets])
    
    dataset.insert(1, "Habitable", -1, True)
    hab_list = habitable_planets["kepoi_name"].tolist()
    for hab_id in hab_list:
        dataset['Habitable'] = np.where(dataset['kepoi_name'] == hab_id, 1, dataset['Habitable'])
    #dataset = dataset.drop_duplicates(subset=['kepoi_name'], keep='first')

    #Shuffling dataset for avoiding habitable concat bias
    dataset = shuffle(dataset)
    dataset.reset_index(inplace=True, drop=True)
    
    return dataset

def get_SVM_Hyper(X_train, y_train):
        param_grid = {'C': np.logspace(-3, 2, 3), 'gamma': np.logspace(-3, 1, 3), 'coef0': np.logspace(-3, 2, 3), 'kernel': ['sigmoid'], 'class_weight': ['balanced']}
        clf = GridSearchCV(svm.SVC(), param_grid, cv = StratifiedKFold(5), refit=True, verbose=1, scoring='accuracy')
        clf.fit(X_train,y_train)
        print(clf.best_params_, "Score: ", clf.best_score_)


def main():
    raw_dataset = dataset_loading()
    dataset, features = dataset_processing(raw_dataset)
    
    y = dataset.Habitable
    X = dataset[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    #X_train, X_test = dataset_normalization(X_train, X_test, 'minmax')
    get_SVM_Hyper(X_train, y_train) #C=0.001 Accuracy 0.99, Accuracy 0.976 with positive duplicates, Accuracy 0.058 with Recall 1.0

    C_grid = input("Insert C value: \n")
    coef0_grid = input("Insert coef0 value: \n")
    gamma_grid = input("Insert gamma value: \n")
    clf = svm.SVC(C=C_grid, kernel='sigmoid', coef0=coef0_grid, gamma=gamma_grid, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    planets = raw_dataset.drop(columns=['Habitable'], axis=1)
    
    data_visualization_analysis(planets, features, clf, X_test, y_pred, y_test)

main()
