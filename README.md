# Predicting Human Diseases from Symptom Profiles Using Multi-Class Classification Models

A comprehensive machine learning project that compares multiple classification algorithms for disease prediction based on patient symptoms and medical conditions.

## 📋 Overview

This project implements and compares the performance of various machine learning algorithms to predict diseases based on patient symptoms. The analysis includes exploratory data analysis (EDA), feature engineering with different encoding techniques, feature selection using Chi-Square tests, and model evaluation with detailed metrics.

## 🎯 Features

- **Comprehensive Data Analysis**: Exploratory data analysis with visualization of feature distributions and correlations
- **Multiple Encoding Strategies**: Implementation of both Label Encoding and One-Hot Encoding
- **Feature Selection**: Chi-Square test for identifying the most relevant features
- **Model Comparison**: Evaluation of 9 different machine learning algorithms
- **Detailed Metrics**: Accuracy, precision, recall, F1-score, and confusion matrices for each model
- **Optimization**: Performance comparison using all features vs. top 3 features

## 🤖 Machine Learning Models Evaluated

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Gradient Boosting Machine (GBM)**
4. **XGBoost Classifier**
5. **K-Nearest Neighbors (KNN)**
6. **Support Vector Machine (SVM)**
7. **Logistic Regression**
8. **Neural Network (MLP)**
9. **Naive Bayes**

## 📊 Dataset Features

The dataset includes the following features:
- **Discharge**: Patient discharge information
- **Feelings and Urge**: Patient-reported feelings and urges
- **Pain and Infection**: Pain levels and infection status
- **Physical Conditions**: Physical examination findings
- **Critical Feelings**: Critical symptom indicators
- **Critical**: Critical condition status
- **Disease**: Target variable (disease diagnosis)

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and tools
- **XGBoost**: Gradient boosting framework
- **SciPy**: Statistical functions

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/disease-prediction-ml-comparison.git
cd disease-prediction-ml-comparison
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook disease_prediction_ml_comparison.ipynb
```

2. Run all cells sequentially to:
   - Load and explore the data
   - Perform feature engineering
   - Train and evaluate all models
   - Compare results with feature selection

## 📈 Methodology

### 1. Data Preprocessing
- Load data from Google Drive
- Exploratory data analysis with visualizations
- Handle missing values and data types

### 2. Feature Engineering
- **Label Encoding**: Encode categorical features as integers
- **One-Hot Encoding**: Create binary columns for categorical values

### 3. Feature Selection
- Chi-Square test to identify feature importance
- Select top 3 most relevant features for optimization

### 4. Model Training & Evaluation
- Split data into training (70%) and testing (30%) sets
- Train multiple classifiers
- Evaluate using accuracy, precision, recall, F1-score
- Generate confusion matrices

### 5. Results Comparison
- Compare all models with full features
- Compare optimized models with top 3 features
- Identify the best-performing algorithm

## 📊 Results

The notebook includes detailed classification reports and confusion matrices for each model. Key findings include:
- Performance metrics for all 9 algorithms
- Feature importance analysis
- Model comparison with full vs. reduced feature sets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Dataset source: Kaggle [https://www.kaggle.com/datasets/follynamylus/diseases-prediction-dataset]
- Course: CSE437 (if this is a course project)
- Special thanks to contributors and reviewers

## 📞 Contact

For questions or feedback, please open an issue or contact [aabuyousufraj@gmail.com]

---

**Note**: Make sure to update the data source link in the notebook if you're using a different dataset location.

