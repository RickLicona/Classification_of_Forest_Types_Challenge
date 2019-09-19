# Classification of Forest Types Challenge
Classification of Forest Types. It is a competition of the platform of Google named Kaggle. 

The techniques used for this competition involve Feature Engineering and Stacking. 

#### Stacking

The classifiers implemented in stacking with its respective parameters  are:

* Random Forest (meta_classifier)
```python
rf_clf = RandomForestClassifier(n_estimators=400,
                                min_samples_leaf=1,
                                verbose=0,
                                random_state=random_state,
                                n_jobs=n_jobs)
```

* Random Forest
```python
rf2_clf = RandomForestClassifier(n_estimators=719,
                                 max_features=0.3,
                                 max_depth=464,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 bootstrap=False,
                                 random_state=42,
                                 n_jobs=-1)
```

* ExtraTreesClassifier
```python
ex_cls = ExtraTreesClassifier(n_estimators=700, criterion='entropy', min_samples_split=3, random_state=42,
                              max_features=0.3,
                              max_depth=464,
                              min_samples_leaf=1,
                              n_jobs=-1)
 ```         
 
#### Feature Engineering
We made basic math operations (e.g., sum, subtraction, division, and mean) to generate new features with the next features:

```python
X['Horizontal_Distance_To_Hydrology']
 ```
 
 ```python
 X['Horizontal_Distance_To_Fire_Points']
  ```

 ```python
 X['Horizontal_Distance_To_Roadways']
  ```
  
The features with a weak correlation with the rest of the features that we dropped  are:     
 ```python
 X.drop(['Soil_Type7', 'Soil_Type15'], axis=1, inplace=True)
 ```            
 