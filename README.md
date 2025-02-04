# Credit Card Fraud Detection  

This repository contains the analyses for the Kaggle dataset **[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**.  

<p align="center">
<img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/readme_image.jpg" width="80%" alt="Readme Image">
</p>  

<br>

My main goal was to improve the score on the test dataset by preprocessing the data and optimizing hyperparameters using Optuna.  

The evaluation metric used was `Average Precision`, as it is well-suited for highly imbalanced datasets, such as credit card fraud detection (Borgne et al., 2022). The machine learning methods applied were `Logistic Regression`, `Random Forest`, and `XGBoost`. Logistic Regression was chosen for its simplicity, while Random Forest and XGBoost are well-suited for this type of analysis (Borgne et al., 2022).  

## Key Findings  
- The choice of preprocessing strategy improved the test dataset score by **1.1% to 30%**, depending on the machine learning method (Figure 1).  
- Hyperparameter optimization improved the test dataset score by **1.1% to 2.1%**, depending on the machine learning method (Figure 2).  

<div style="display: flex; justify-content: center; gap: 20px;">
    <img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/preprocessor.jpg" width="48%" alt="Preprocessing Results">
    <img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/hp_optimization.jpg" width="48%" alt="Hyperparameter Optimization Results">
</div>  

<br>

Below is a succinct description of the steps taken in this project.

## [Exploratory Data Analysis and Data Engineering](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/01_ah_eda.ipynb)  
At this stage, the following procedures were performed: 

1. **Sanity Check**  
   - Conducted an initial inspection for duplicate entries and missing values across features.

2. **Numeric Features**  
   - Inspected the normality of the features, pairwise correlations, and the presence of outliers.

3. **Feature-Target Relationships**  
   - Examined the association between features and the target variable.  

4. **Target Variable Analysis**  
   - Observed that fraud cases were relatively evenly distributed over time.  
   - Decided to use the first day's data for training the models and the second day's data for testing.

5. **Time Feature**  
   - Created a binary feature representing periods of low (hours 1–6) and high (hours 0, 7–23) transaction amounts. This improved the test dataset score by **0.7%** ([see details](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/ah_appendix.ipynb)).  

6. **Feature Transformation**  
   - Explored whether PowerTransformer or QuantileTransformer would be the best transformation strategy.  

## [Machine Learning](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/02_ah_model.ipynb)  
In this phase, I implemented the following steps:  

1. **Preprocessing**  
   - Determined the best preprocessing strategy for each machine learning method.

2. **Handling Imbalance**  
   - Used the hybrid over- and under-sampling **SMOTE-TOMEK** method to confirm that Average Precision is robust against class imbalance.

3. **Hyperparameter Optimization**  
   - Utilized the **Optuna** framework to optimize hyperparameters, improving the test dataset scores over the default **Scikit-Learn** and **XGBoost** hyperparameters.  

## Tools and Technologies  
- **Libraries**: Imbalanced-learn, Matplotlib, NumPy, Optuna, Pandas, Seaborn, Scikit-Learn, XGBoost  

## Project organization

```
├── .gitignore                         <- Files and directories to be ignored by Git  
│  
├── environments.yml                   <- Requirements file to reproduce the analysis environment  
│  
├── LICENSE                            <- License type  
│  
├── README.md                          <- Main README explaining the project  
│  
├── data                               <- Project data files  
│   ├── logistic_regression_round1.db  <- Optuna study for Logistic Regression  
│   ├── random_forest_round1.db        <- Optuna study for Random Forest with multiple hyperparameters  
│   ├── random_forest_round2.db        <- Optuna study for Random Forest with selected hyperparameters  
│   ├── xgboost_round1.db              <- Optuna study for XGBoost with multiple hyperparameters  
│   ├── xgboost_round2.db              <- Optuna study for XGBoost with selected hyperparameters  
│  
├── models                             <- Trained and serialized models, model predictions, or model summaries  
│   ├── best_model.pkl                 <- Deployed model  
│  
├── notebooks                          <- Jupyter notebooks  
│   ├── 01_ah_EDA.ipynb                <- Exploratory data analysis  
│   ├── 02_ah_MODEL.ipynb              <- Machine learning approach  
│   ├── ah_appendix.ipynb              <- Supplementary analyses and graphs  
│  
├── src                                <- Source code used in this project  
│   ├── __init__.py                    <- Makes this a Python module  
│   ├── config.py                      <- Basic project configuration  
│   ├── data_size_optimization.py      <- Script to optimize dataset size  
│   ├── eda.py                         <- Script for exploratory data analysis and visualizations  
│   ├── models.py                      <- Script for GridSearchCV  
│  
├── references                         <- Data dictionaries, manuals, and other explanatory materials  
│   ├── 01_data_dictionary.md          <- Description of the dataset as presented on Kaggle  
│  
├── reports                            <- Generated analyses in HTML, PDF, LaTeX, etc., and results  
│   └── images                         <- Images used in the project  
│      ├── hp_optimization.jpg         <- Graph showing results from hyperparameter optimization  
│      ├── preprocessor.jpg            <- Graph showing results from preprocessing analysis  
│      ├── readme_image.jpg            <- Image to illustrate the README  

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

## References

Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection—Practical Handbook. Université Libre de Bruxelles.