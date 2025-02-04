# Credit Card Fraud Detection

This repository contains the analyses for the Kaggle's **[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**. 

<img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/readme_image.jpg" width="80%" alt="Readme Image">
credit: FreePix.uk

My main goal was to improve the score on the test dataset by preprocessing the data and optimizing the hyperparameter using Optuna. The score adopted was `Average Precision`, as it is appropriate for cases with high imbalanced dataset, such in Credit Card Fraud Detection cases (Borgne et al., 2022). The machine learning methodss were `Logistic Regression`, `Randon Forest` and `XGBooster`. The first for its simplicity and the other two because they are well suitated methods for this kind of analysis (Borgne et al., 2022).  

## Key Findings:
- The preprocessing choice can improve from 3% to 29% the score on the test dataset, depending on the machine learning method (Figure 1).
- Hyperparameter optimization improved in 1% to 2% the score on the test dataset, depending on the machine learning method (Figure 2).



Below I succinctly describe the steps developed. 

## [Adding Latitude and Longitude to the Dataset](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/01_ah_merging_datas.ipynb)  
- Combined the latitude and longitude information available in [R's tidymodels](https://www.tmwr.org/ames) dataset with Kaggle's dataset, which brought a slight improvement in the score (Table 1).
- Produced maps (see [here](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_1.ipynb)).

## [Exploratory Data Analysis and Data Engineering](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/02_ah_EDA.ipynb)  
At this stage, I went through several procedures:  
1. **Sanity check**
- Performed an initial inspection for duplicate entries, standardized text data, and analyzed the distribution of missing values across features.
- Dropped categorical features that lacked a second representative category in the training dataset, resulting in a ~6% reduction in feature set and a slight score improvement (Table 1). 

2. **Missing values treatment**
- Although AutoGluon handles missing values by default, I implemented a custom approach, which led to a performance improvement of approximately 1.1% compared to the AutoGluon default handling (Table 1).
  
3. **Categorical features**
- Tested two approaches: combining underrepresented categories to increase sample size and introducing new features. Despite these efforts, neither approach resulted in a beter score on the test dataset (Table 2).

4. **Numeric features**
- Experimented with creating delta features based on year differences, introducing a seasonal feature for the time of sale, and features related with the area of the houses. Despite these efforts, neither approach resulted in a beter score on the test dataset (Table 2).

5. **Numeric vs. non-numeric features**
- Explored an alternative encoding method for ordinal features (see [here](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_2.ipynb)), given a clear relationships between some of those features and the target. However, my approaches did not improve the score on the test dataset (Table 2).
  
6. **Outliers**
- Evaluated the impact of removing extreme outliers from the dataset. This method did not result in improved score on the test dataset (Table 2).
  
7. **New feature based on location**
- Introduced a new feature that represented the median sale price of the 100 closest houses, with the number of neighboring houses determined through experimentation (values tested: 5, 50, 100, 150, and 200). This feature contributed to a meaningful improvement in model performance, yielding an increase of ~1.2% over the previous best approach (Table 1).

8. **New feature based on economic data**
- Tested two economic indexes related to housing prices, but these features did not improve model performance (Table 2).

9. **Interaction between original features**
- Created interaction features by combining pairs of the top ten most important numeric/ordinal features (e.g., 'GrLivArea' * 'OverallQual'), based on AutoGluon’s feature importance ranking. This approach further improved the score by ~0.25% (Table 1).  

## [Machine Learning](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/03_ah_MODEL.ipynb)  
- I used AutoGluon to develop the machine learning analyses. All analyses were done using the `presets = 'good_quality'` and `time_limit = 30 min`.

### Table 2: Proposed Treatments That Did Not Improve the Scores in the Test Dataset  

| **Treatments** |  
|:--------------:|  
| Combining underrepresented classes |  
| Grouping neighborhoods into larger areas |  
| Defining the seasons of the year |  
| Defining deltas to represent features related to construction, remodeling, and selling year | 
| Defining total area, finished are and high quality area of the houses |
| OrdinalEncoder |  
| Removing highly discrepant outliers |  
| Case-Shiller U.S. National Home Price Index |  
| All-Transactions House Price Index for Ames, IA |  

## Tools and Technologies:
- **Libraries**: Imblearn, Matplotlib, Numpy, Optuna, Pandas, Seaborn, Scikit-Learn, Xgboost


## Project organization

```
├── .gitignore                         <- Files and directories to be ignored by Git
|
├── environments                       <- Requirements files to reproduce the analysis environment
|   ├── autogluon_environment.yml      <- Environment for running '03_ah_MODEL.ipynb' 
|   ├── eda_environment.yml            <- Environment for running '01_ah_merging_datas.ipynb', '02_ah_EDA.ipynb', and 'ah_appendix_2.ipynb' 
|   ├── maps_environment.yml           <- Environment for running 'ah_appendix_1.ipynb'
|
├── LICENSE                            <- License type 
|
├── README.md                          <- Main README explaining the project
|
├── data                               <- Project data files
|   ├── ames_dataset.csv               <- Original dataset from R's tidymodels
|   ├── ATNHPIUS11180Q.csv             <- Data for Ames' House Price Index
|   ├── clean_data.csv                 <- Dataset prepared for machine learning analysis
|   ├── CSUSHPINSA.csv                 <- Data for the USA's Case-Shiller Index
|   ├── original_plus_lat_lon.csv      <- Kaggle dataset with added longitude and latitude from tidymodels
|   ├── test.csv                       <- Original test dataset from Kaggle
|   ├── train.csv                      <- Original train dataset from Kaggle
|
├── models                             <- Trained and serialized models, model predictions, or model summaries
|
├── notebooks                          <- Jupyter notebooks
|   ├── 01_ah_merging_datas.ipynb      <- Adding latitude and longitude information to Kaggle's dataset
|   ├── 02_ah_EDA.ipynb                <- Exploratory data analysis
|   ├── 03_ah_MODEL.ipynb              <- Machine learning approach
|   ├── ah_appendix_1.ipynb            <- Creating maps
|   ├── ah_appendix_2.ipynb            <- Exploring an alternative dataset transformation
|   └── src                            <- Source code used in this project
|      ├── __init__.py                 <- Makes this a Python module
|      ├── auxiliaries.py              <- Scripts to compute nearest houses and median sale prices
|      ├── config.py                   <- Basic project configuration
|      ├── data_size_optimization.py   <- Scripts to optimize the dataset size
|      ├── eda.py                      <- Scripts for exploratory data analysis and visualizations
|
├── references                         <- Data dictionaries, manuals, and other explanatory materials
|   ├── data_description.txt           <- Description of the dataset as presented on Kaggle
|
├── reports                            <- Generated analyses in HTML, PDF, LaTeX, etc., and results
|   ├── best_kaggle_prediction.csv     <- Best prediction from Kaggle competition
│   └── images                         <- Images used in the project
|      ├── sale_prices_train_test_plot.png  <- Map showing houses with and without prices 
|      ├── sale_prices.png                  <- Map displaying house prices
```

## Contributing
All contributions are welcome!

### Issues
Submit issues for:
- Recommendations or improvements
- Additional analyses or models
- Feature enhancements
- Bug reports

### Pull Requests
- Open an issue before starting work.
- Fork the repository and clone it.
- Create a branch and commit your changes.
- Push your changes and open a pull request for review.

### References

Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection—Practical Handbook. Université Libre de Bruxelles.