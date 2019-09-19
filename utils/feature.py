import numpy as np


def feature_eng(X):
    X.drop(['Soil_Type7', 'Soil_Type15'], axis=1, inplace=True)

    X['HF1'] = X['Horizontal_Distance_To_Hydrology'] + X['Horizontal_Distance_To_Fire_Points']
    X['HF2'] = abs(X['Horizontal_Distance_To_Hydrology'] - X['Horizontal_Distance_To_Fire_Points'])
    X['HR1'] = abs(X['Horizontal_Distance_To_Hydrology'] + X['Horizontal_Distance_To_Roadways'])
    X['HR2'] = abs(X['Horizontal_Distance_To_Hydrology'] - X['Horizontal_Distance_To_Roadways'])
    X['FR1'] = abs(X['Horizontal_Distance_To_Fire_Points'] + X['Horizontal_Distance_To_Roadways'])
    X['FR2'] = abs(X['Horizontal_Distance_To_Fire_Points'] - X['Horizontal_Distance_To_Roadways'])
    X['ele_vert'] = X.Elevation - X.Vertical_Distance_To_Hydrology

    X['slope_hyd'] = (X['Horizontal_Distance_To_Hydrology'] ** 2 + X['Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    X.slope_hyd = X.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)  # remove infinite value if any

    # Mean distance to Amenities
    X['Mean_Amenities'] = (
                                      X.Horizontal_Distance_To_Fire_Points + X.Horizontal_Distance_To_Hydrology + X.Horizontal_Distance_To_Roadways) / 3
    # Mean Distance to Fire and Water
    X['Mean_Fire_Hyd'] = (X.Horizontal_Distance_To_Fire_Points + X.Horizontal_Distance_To_Hydrology) / 2

    X['Mean_HF1'] = X.HF1 / 2
    X['Mean_HF2'] = X.HF2 / 2
    X['Mean_HR1'] = X.HR1 / 2
    X['Mean_HR2'] = X.HR2 / 2
    X['Mean_FR1'] = X.FR1 / 2
    X['Mean_FR2'] = X.FR2 / 2

    # NEW
    X['HF_SUM'] = X['Mean_HF1'] + X['Mean_HF2']
    X['HR_SUM'] = X['Mean_HR1'] + X['Mean_HR2']
    X['FR_SUM'] = X['Mean_FR1'] + X['Mean_FR2']

    return X
