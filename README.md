# Credit Card Fraud Detection  
[Go to German Version](#german-version)

This repository contains the analyses for the Kaggle dataset **[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**. See the summary report [here](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/summary_report.md).



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

---

## Tools and Technologies  
- **Libraries**: Imbalanced-learn, Matplotlib, NumPy, Optuna, Pandas, Seaborn, Scikit-Learn, XGBoost  

---

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
|   ├── summary_report.md              <- Summary report
│   └── images                         <- Images used in the project  
│       ├── hp_optimization.jpg        <- Graph showing results from hyperparameter optimization  
│       ├── preprocessor.jpg           <- Graph showing results from preprocessing analysis  
│       ├── readme_image.jpg           <- Image to illustrate the README  

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

# German Version

# Kreditkartenbetrugserkennung  

Dieses Repository enthält die Analysen für den Kaggle-Datensatz **[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**. Siehe den zusammenfassenden Bericht [hier](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/summary_report.md)

<p align="center">
<img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/readme_image.jpg" width="80%" alt="Readme Image">
</p>  

<br>

Mein Hauptziel war es, die Bewertung des Testdatensatzes durch die Vorverarbeitung der Daten und die Optimierung von Hyperparametern mit Optuna zu verbessern.  

Als Bewertungsmetrik wurde `Average Precision` verwendet, da sie gut für stark unausgeglichene Datensätze geeignet ist, wie z. B. die Erkennung von Kreditkartenbetrug (Borgne et al., 2022). Die angewendeten maschinellen Lernmethoden waren `Logistische Regression`, `Random Forest` und `XGBoost`. Die Logistische Regression wurde aufgrund ihrer Einfachheit gewählt, während Random Forest und XGBoost gut für diese Art der Analyse geeignet sind (Borgne et al., 2022).  

## Wichtige Erkenntnisse  
- Die Wahl der Vorverarbeitungsstrategie verbesserte die Bewertung des Testdatensatzes um **1,1 % bis 30 %**, abhängig von der maschinellen Lernmethode (Abbildung 1).  
- Die Hyperparameteroptimierung verbesserte die Bewertung des Testdatensatzes um **1,1 % bis 2,1 %**, abhängig von der maschinellen Lernmethode (Abbildung 2).  

<div style="display: flex; justify-content: center; gap: 20px;">
    <img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/preprocessor.jpg" width="48%" alt="Vorverarbeitungsergebnisse">
    <img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/hp_optimization.jpg" width="48%" alt="Hyperparameter-Optimierungsergebnisse">
</div>  

<br>

Nachfolgend finden Sie eine kurze Beschreibung der in diesem Projekt durchgeführten Schritte.

## [Explorative Datenanalyse und Datenengineering](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/01_ah_eda.ipynb)  
In dieser Phase wurden die folgenden Verfahren durchgeführt: 

1. **Plausibilitätsprüfung**  
   - Durchführung einer ersten Überprüfung auf doppelte Einträge und fehlende Werte in den Merkmalen.

2. **Numerische Merkmale**  
   - Überprüfung der Normalverteilung der Merkmale, paarweise Korrelationen und das Vorhandensein von Ausreißern.

3. **Merkmals-Ziel-Beziehungen**  
   - Untersuchung der Beziehung zwischen den Merkmalen und der Zielvariablen.  

4. **Analyse der Zielvariablen**  
   - Feststellung, dass Betrugsfälle relativ gleichmäßig über die Zeit verteilt waren.  
   - Entscheidung, die Daten des ersten Tages für das Training der Modelle und die Daten des zweiten Tages für das Testen zu verwenden.

5. **Zeitmerkmal**  
   - Erstellung eines binären Merkmals, das Perioden mit niedrigen (Stunden 1–6) und hohen (Stunden 0, 7–23) Transaktionsbeträgen darstellt. Dies verbesserte die Bewertung des Testdatensatzes um **0,7 %** ([siehe Details](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/ah_appendix.ipynb)).  

6. **Merkmals-Transformation**  
   - Untersuchung, ob PowerTransformer oder QuantileTransformer die beste Transformationsstrategie darstellen.  

## [Maschinelles Lernen](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/02_ah_model.ipynb)  
In dieser Phase wurden die folgenden Schritte implementiert:  

1. **Vorverarbeitung**  
   - Bestimmung der besten Vorverarbeitungsstrategie für jede maschinelle Lernmethode.

2. **Umgang mit Unausgeglichenheit**  
   - Verwendung der hybriden Über- und Unterabtastungsmethode **SMOTE-TOMEK**, um zu bestätigen, dass Average Precision robust gegenüber Klassenungleichgewichten ist.

3. **Hyperparameter-Optimierung**  
   - Verwendung des **Optuna**-Frameworks zur Optimierung von Hyperparametern, wodurch die Bewertungen des Testdatensatzes im Vergleich zu den Standard-Hyperparametern von **Scikit-Learn** und **XGBoost** verbessert wurden.  

## Tools und Technologien  
- **Bibliotheken**: Imbalanced-learn, Matplotlib, NumPy, Optuna, Pandas, Seaborn, Scikit-Learn, XGBoost  

## Projektorganisation
```
├── .gitignore                         <- Dateien und Verzeichnisse, die von Git ignoriert werden sollen  
│  
├── environments.yml                   <- Anforderungsdatei zur Reproduktion der Analyseumgebung  
│  
├── LICENSE                            <- Lizenztyp  
│  
├── README.md                          <- Haupt-README, die das Projekt erklärt  
│  
├── data                               <- Projektdateien  
│   ├── logistic_regression_round1.db  <- Optuna-Studie für Logistische Regression  
│   ├── random_forest_round1.db        <- Optuna-Studie für Random Forest mit mehreren Hyperparametern  
│   ├── random_forest_round2.db        <- Optuna-Studie für Random Forest mit ausgewählten Hyperparametern  
│   ├── xgboost_round1.db              <- Optuna-Studie für XGBoost mit mehreren Hyperparametern  
│   ├── xgboost_round2.db              <- Optuna-Studie für XGBoost mit ausgewählten Hyperparametern  
│  
├── models                             <- Trainierte und serialisierte Modelle, Modellvorhersagen oder Modellzusammenfassungen  
│   ├── best_model.pkl                 <- Bereitgestelltes Modell  
│  
├── notebooks                          <- Jupyter-Notebooks  
│   ├── 01_ah_EDA.ipynb                <- Explorative Datenanalyse  
│   ├── 02_ah_MODEL.ipynb              <- Ansatz des maschinellen Lernens  
│   ├── ah_appendix.ipynb              <- Ergänzende Analysen und Grafiken  
│  
├── src                                <- In diesem Projekt verwendeter Quellcode  
│   ├── __init__.py                    <- Macht dies zu einem Python-Modul  
│   ├── config.py                      <- Grundlegende Projektkonfiguration  
│   ├── data_size_optimization.py      <- Skript zur Optimierung der Datensatzgröße  
│   ├── eda.py                         <- Skript für explorative Datenanalyse und Visualisierungen  
│   ├── models.py                      <- Skript für GridSearchCV  
│  
├── references                         <- Datenwörterbücher, Handbücher und andere erklärende Materialien  
│   ├── 01_data_dictionary.md          <- Beschreibung des Datensatzes, wie auf Kaggle präsentiert  
│  
├── reports                            <- Generierte Analysen in HTML, PDF, LaTeX usw. und Ergebnisse  
|   ├── summary_report.md              <- Zusammenfassender Bericht
│   └── images                         <- Im Projekt verwendete Bilder  
│       ├── hp_optimization.jpg        <- Grafik mit den Ergebnissen der Hyperparameter-Optimierung  
│       ├── preprocessor.jpg           <- Grafik mit den Ergebnissen der Vorverarbeitungsanalyse  
│       ├── readme_image.jpg           <- Bild zur Veranschaulichung der README  
```

## Mitwirkung
Alle Beiträge sind willkommen!

### Probleme
Reichen Sie Probleme ein für:
- Empfehlungen oder Verbesserungen
- Zusätzliche Analysen oder Modelle
- Funktionserweiterungen
- Fehlermeldungen

### Pull Requests
- Öffnen Sie ein Problem, bevor Sie mit der Arbeit beginnen.
- Forken Sie das Repository und klonen Sie es.
- Erstellen Sie einen Branch und committen Sie Ihre Änderungen.
- Pushen Sie Ihre Änderungen und öffnen Sie einen Pull Request zur Überprüfung.

## Referenzen

Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection—Practical Handbook. Université Libre de Bruxelles.
