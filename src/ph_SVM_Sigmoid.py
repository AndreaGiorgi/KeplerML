"""
@authors: Andrea Giorgi and Gianluca De Angelis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA, PCA
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

## Results visualization
# Visualize prediction results in order to analize model behaviour and performances
# 1. Plot predicted habitable planets 
# 2. Plot predicted habitable planets which respects habitable parameters
#    Parameters are: 
#       1. Planet-Star distance between [0.85, 1.7] AU
#       2. Temperature between [200,600] Kelvin
#       3. Planet radius between [0.5, 3.3] Earth radius


def data_visualization_analysis(planets, features, predictions):
    
    X_distance_from_parent_star = []
    Y_surface_temprature = []
    S_planet_radius = []
    colors = []
    color_space = 255*255*255
    
    planets = planets.fillna(0)
    
    total_distance = 0
    total_temperature = 0
    minimum_temperature = 0
    maximum_temperature = 0
    number_of_habitable_planets = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            habitable_planet_koi = planets.iloc[i, 2] #kepoi_name
            planet_temperature = planets.iloc[i, 58] - 273.15  #koi_teq in Celsius 
            if planet_temperature > maximum_temperature:
                maximum_temperature = planet_temperature
            elif planet_temperature < minimum_temperature:
                minimum_temperature = planet_temperature
            total_temperature += planet_temperature
            planet_radius = planets.iloc[i, 49] #koi_prad
            planet_star_distance = planets.iloc[i, 64] #koi_dor
            total_distance += planet_star_distance
            number_of_habitable_planets += 1
            print('Predicted Habitable planet koi = ',habitable_planet_koi, ", Equilibrium Temperature in Celsius = ", planet_temperature, ", Planet radius (Earth) = ", planet_radius)         
            X_distance_from_parent_star.append(planet_star_distance)
            Y_surface_temprature.append(planet_temperature)
            S_planet_radius.append(planet_radius)
            colors.append(np.random.randint(color_space))
    
    print("Features used were: ", features)
    print("Number of habitable planets detected: " , number_of_habitable_planets)
    mean_distance = total_distance/number_of_habitable_planets
    print('Mean distance of habitable planets: ' , mean_distance)
    var_distance = np.sqrt(np.sum(np.square(X_distance_from_parent_star - mean_distance))/number_of_habitable_planets)
    print('Standard deviation of distance of habitable planets: ' , var_distance)
    
    mean_temp = total_temperature/number_of_habitable_planets
    print('Minimum temperature of habitable planet: ' , minimum_temperature, " Celsius")
    print('Maximum temperature of habitable planet: ' , maximum_temperature, " Celsius")   
    print('Mean temperature of habitable planets: ' , mean_temp, " Celsius")
    var_temp = np.sqrt(np.sum(np.square(Y_surface_temprature - mean_temp))/number_of_habitable_planets)
    print('Standard deviation temperature of habitable planets ' , var_temp)
    
    plt.scatter(X_distance_from_parent_star, Y_surface_temprature, s = S_planet_radius, c = colors)
    plt.xlabel('Distance from parent star')
    plt.ylabel('Planetary Equilibrium Temperature in Celsius')
    plt.show()
    
    X_distance_from_parent_star_verified = []
    Y_surface_temprature_verified = []
    S_planet_radius_verified = []
    colors_verified = []
    
    minimum_temperature = 0
    maximum_temperature = 0    
    number_of_confirmed_habitable_planets = 0
    
    for i in range(len(predictions)):
        if predictions[i] == 1:
            habitable_planet_koi = planets.iloc[i, 2] #kepoi_name
            planet_temperature = planets.iloc[i, 58] #koi_teq in Celsius 
            planet_temperature_celsius = planets.iloc[i, 58] - 273.15  #koi_teq in Celsius 
            planet_radius = planets.iloc[i, 49] #koi_prad
            planet_star_distance = planets.iloc[i, 64] #koi_dor
            number_of_confirmed_habitable_planets += 1
            if 183 <= planet_star_distance <= 460 and 200 <= planet_temperature <= 600 and 0.5 <= planet_radius <= 3.3:
                print('Predicted "confirmed" Habitable planet koi = ',habitable_planet_koi, ", Equilibrium Temperature in Celsius = ", planet_temperature_celsius, ", Planet radius (Earth) = ", planet_radius)         
                X_distance_from_parent_star_verified.append(planet_star_distance)
                Y_surface_temprature_verified.append(planet_temperature_celsius)
                S_planet_radius_verified.append(planet_radius)
                colors_verified.append(np.random.randint(color_space))

    plt.scatter(X_distance_from_parent_star_verified, Y_surface_temprature_verified, s = S_planet_radius_verified, c = colors_verified)
    plt.xlabel('Distance from parent star')
    plt.ylabel('Planetary Equilibrium Temperature in Celsius')
    plt.show()           
    
## Dataset normalization algorihtms
# 1. StandardScaling: Standardize features by removing the mean and scaling to unit variance 
# 2. MinMax Scaling: Transform features by scaling each feature to a given range.

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

## Principal component analysis (PCA). 
# Linear dimensionality reduction using Singular Value Decomposition SVD of the data to project it to a lower dimensional space. 
# The input data is centered but not scaled for each feature before applying the SVD.

def get_PCA(dataset):
    
    PCATransformer = PCA(n_components = 6, whiten = 'True', svd_solver = 'full')
    data = PCATransformer.fit_transform(dataset)
    
    return data

## Kernel Principal component analysis (KPCA). 
# Non-linear dimensionality reduction through the use of kernels, here we use sigmoid in order to be consistent with SVM kernel

def get_KPCA(dataset):
    
    KPCAtransformer = KernelPCA(n_components = 6, kernel='sigmoid', eigen_solver = 'arpack', random_state = 42)
    data = KPCAtransformer.fit_transform(dataset)
     
    return data

## Test set preprocessing (ETL Pipeline)
# 1. Loads dataset with planetary features 
# 2. Checks for missing data and drops rows with missing data


def test_set_processing(dataset):
        
    planetary_stellar_features = ["koi_period", "koi_ror", "koi_srho", "koi_prad", 
                                  "koi_sma", "koi_teq", "koi_insol", "koi_dor", "koi_count", 
                                  "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass"]
    dataset = dataset[planetary_stellar_features]
    
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
        
    dataset = dataset.dropna()
    
    return dataset

## Training set preprocessing (ETL Pipeline)
# 1. Loads dataset with planetary features 
# 2. Checks for missing data and fillis rows with missing data with default 0 value 
# 3. Calculates and plot features correlation for analisys purpose 

def training_set_processing(dataset):
    
    ## Drop all columns that are not listed in Planetary columns
    
    planetary_stellar_features = ["Habitable", "koi_period", "koi_ror", "koi_srho", "koi_prad", 
                                  "koi_sma", "koi_teq", "koi_insol", "koi_dor", "koi_count", 
                                  "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass"]
    dataset = dataset[planetary_stellar_features]
    
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
    
    correlation = dataset.corr()
    correlation_target = abs(correlation["Habitable"])
    print(correlation_target)
    sns.heatmap(correlation, 
            xticklabels=correlation.columns.values,
            yticklabels=correlation.columns.values)
    plt.show()

    return dataset

## Dataset loading (ETL pipeline) 
# 1. Load non habitable and habitable planets data 
# 2. Concatenate them in training set and add prediction label Habitabile, default value -1 [Non Habitable]
# 3. Set Habitable label to 1 for each confirmed habitable planet 
# 4. Shuffle dataset in order to reduce order dependency 
# 5. Load and shuffle test data, taken from cumulative keplero data

def datasets_loading():
    
    non_habitable = pd.read_csv('data/non_habitable_planets_confirmed_detailed_list.csv')
    habitable_planets = pd.read_csv('data/habitable_planets_detailed_list.csv')
    training_set = pd.concat([non_habitable, habitable_planets])

    training_set.insert(1, "Habitable", -1, True)
    hab_list = habitable_planets["kepoi_name"].tolist()
    for hab_id in hab_list:
        training_set['Habitable'] = np.where(training_set['kepoi_name'] == hab_id, 1, training_set['Habitable'])

    training_set = shuffle(training_set)
    training_set.reset_index(inplace=True, drop=True)
    print("Test set shape: ")
    print(training_set.shape, '\n')
    
    test_set = pd.read_csv('data/cumulative_new_data.csv')
    print("Test set shape: ")
    print(test_set.shape, '\n')
    test_set.shape
    test_set = shuffle(test_set, random_state = 0)
    test_set.reset_index(inplace=True, drop=True)
    
    return training_set, test_set

## Support Vector Machine Hyperparameters estimation 
# 1. Defines the paramaters grid 
# 2. Use GridSearchCV to estimate the best parameters, it values them in order to achive highest accuracy 
# 3. Show the best parameters and it allows to set manually svm parameters, based on estimated ones 
# 4. Initialize optimized model and it fits the model on X_Train and y_train

def get_SVM_Hyper(X_train, y_train):
    
        param_grid = {'C': np.logspace(-3, 1, 3), 'gamma': np.logspace(-3, 1, 3), 'coef0': np.logspace(-3, 1, 3), 
                      'kernel': ['sigmoid'], 'class_weight': ['balanced']}
        params_estimator = GridSearchCV(svm.SVC(), param_grid, cv = StratifiedKFold(10), refit=True, verbose=1, scoring = 'recall')
        params_estimator.fit(X_train,y_train)
        print(params_estimator.best_params_, "\n Recall score with estimated hyperparameters: ", params_estimator.best_score_)
        
        C_grid = float(input("Insert C value: \n"))
        coef0_grid = float(input("Insert coef0 value: \n"))
        gamma_grid = float(input("Insert gamma value: \n"))
    
        model = svm.SVC(C=C_grid, kernel='sigmoid', coef0=coef0_grid, gamma=gamma_grid, class_weight='balanced')
        model.fit(X_train, y_train)
        
        return model
 
 ## Train and test set initializer 
 # 1. Loads prediction label and train/test sets 
 # 2. Applies Sequential Feature Selection algorithm in order to extract most important features 
 # 3. Normalize X_train and X_test using standard scaling or MinMax scaling 
 # 4. Applies dimensionality reduction technique PCA or KPCA
 
def get_train_test(train, test, normalization, dim_reduction):
    
    y_train = train.Habitable
    X_train = train.drop('Habitable', 1)
    X_test = test
    
    sfs = SequentialFeatureSelector(estimator=svm.SVC(kernel='sigmoid'), cv=StratifiedKFold(10), direction='backward')
    sfs.fit(X_train, y_train)
    selected_features= X_train.columns[(sfs.get_support())]
    X_train = X_train[selected_features]
    
    ## Normalization with Standard Scaling or MinMax scaling
    if normalization is not None:
        X_train, X_test = dataset_normalization(train, test, normalization)
    
    ## Principal Component Analysis PCA or Kernel PCA KPCA
    if dim_reduction == 'PCA':
        X_train = get_PCA(X_train)
        X_test = get_PCA(X_test)
    elif dim_reduction == 'KPCA':
        X_train = get_KPCA(X_train)
        X_test = get_KPCA(X_test)
        
    return X_train, X_test, selected_features
         
def prediction_pipeline():
    
    ## ETL
    raw_training_set, raw_planets_set = datasets_loading()
    training_set = training_set_processing(raw_training_set)
    test_set = test_set_processing(raw_planets_set)
    
    ## Feature selection and X and Y selection
    y_train = training_set.Habitable
    X_train, X_test, features = get_train_test(training_set, test_set, 'standard', 'PCA')
    
    ## SVM Model initialization
    svm_model = get_SVM_Hyper(X_train, y_train)
    
    ## Predict habitable planets
    y_predicted = svm_model.predict(X_test)
    
    ## Results visualization
    data_visualization_analysis(raw_planets_set, features, y_predicted)

if __name__ == "__main__":
    prediction_pipeline()