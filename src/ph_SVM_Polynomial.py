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

# TODO
# 1. Implement ETL pipeline
# 2. Implement Feature selection algorithm
# 3. Implement Dataset normalization with Standard Scaling (StandardScaler)
# 4. Implement SVM model with Polynomial Kernel
# 5. Results visualization

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
            habitable_planet_koi = planets.iloc[i, 2] #kepoi_name: Planet Kepler code
            planet_temperature = planets.iloc[i, 58] - 273.15  #koi_teq: Planet temperature converted from Kelvin to Celsius
            if planet_temperature > maximum_temperature:
                maximum_temperature = planet_temperature
            elif planet_temperature < minimum_temperature:
                minimum_temperature = planet_temperature
            total_temperature += planet_temperature
            planet_radius = planets.iloc[i, 49] #koi_prad: Planet Radius
            planet_star_distance = planets.iloc[i, 64] #koi_dor: Distance from Planet to Star measured in Earth-Sun distance
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
            habitable_planet_koi = planets.iloc[i, 2] #kepoi_name: Planet Kepler code
            planet_temperature = planets.iloc[i, 58] #koi_teq: Planet temperature converted from Kelvin to Celsius
            planet_temperature_celsius = planets.iloc[i, 58] - 273.15  #koi_teq in Celsius 
            planet_radius = planets.iloc[i, 49] #koi_prad: Planet Radius
            planet_star_distance = planets.iloc[i, 64] #koi_dor: Distance from Planet to Star measured in Earth-Sun distance
            
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
    
