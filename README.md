# Smoking Prediction from Body Signals

A machine learning pipeline that predicts whether a person **smokes** from routine health-check data: body measurements (height, weight, waist, eyesight, hearing), blood pressure, and blood and urine tests (cholesterol, triglycerides, hemoglobin, AST/ALT, GTP and more).

Built for the *Machine Learning* exam of the BSc in Artificial Intelligence (University of Pavia), February 2025.

## Results

| Metric (test set) | Value |
|---|---|
| F1-score | **0.774** |
| ROC AUC | **0.807** |
| Accuracy | 0.74 |
| Recall (smokers) | 0.87 |
| Train/test F1 gap | 0.008 (no overfitting) |

The final model is a **Logistic Regression** (L1-regularised) on 10 features chosen by **Sequential Feature Selection**. The model is tuned for high recall on smokers: in a screening setting a false negative costs more than a false positive.

## Dataset

- Source: [Body signal of smoking (Kaggle)](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking)
- `dataset_initial.csv`: the original sample.
- `dataset_noiser.py`: a script that injects missing values and noise so the cleaning steps can be tested realistically.
- `dataset_smoking_habits.csv`: the resulting dataset used in the notebook (about 12.5k rows, 27 columns; target `smoking` = 0/1).

## Methodology

1. **Preprocessing**: target-balance check, removal of rows and columns with too many missing values, removal of near-constant columns, imputation (`SimpleImputer`, `KNNImputer`), scaling and encoding inside a scikit-learn `ColumnTransformer`.
2. **Model selection**: a `RandomizedSearchCV` over full pipelines (sampler → dimensionality reduction → classifier), evaluated with repeated stratified k-fold cross-validation.
   - Samplers: none / Random over-sampling / SMOTE (`imbalanced-learn`)
   - Dimensionality reduction: none / PCA / LDA / Sequential Feature Selection (`mlxtend`)
   - Classifiers: Logistic Regression, Random Forest, SVC, KNN, Perceptron
3. **Refinement**: a second, narrower hyperparameter search around the best configuration.
4. **Evaluation**: classification report, confusion matrix, learning and validation curves, precision–recall threshold analysis, ROC curve.

## Run it

```bash
git clone https://github.com/brusati04/smoking-prediction-body-signals.git
cd smoking-prediction-body-signals
pip install -r requirements.txt
jupyter notebook smoking_prediction.ipynb
```

## Repository structure

```
├── smoking_prediction.ipynb     # full pipeline: EDA → preprocessing → model selection → evaluation
├── dataset_smoking_habits.csv   # dataset used by the notebook
├── dataset_initial.csv          # original Kaggle sample
├── dataset_noiser.py            # adds missing values / noise
└── requirements.txt
```

## Possible extensions

- Gradient-boosting models (XGBoost / LightGBM) and probability calibration
- SHAP values to explain which biomarkers drive the prediction

## License

[MIT](LICENSE) © Lorenzo Brusati
