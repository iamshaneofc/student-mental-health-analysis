# Student Depression Prediction

## Overview

This project aims to predict depression risk among students using machine learning and provide a user-friendly web application for early intervention. By analyzing the "Student Depression Dataset," the project identifies key risk factors such as suicidal thoughts, academic pressure, financial stress, and sleep duration. A Logistic Regression model, optimized for high recall (0.9085) and an F1 score of 0.8576, powers the predictions. The Flask-based web application, styled with Tailwind CSS, allows users to input student data and receive depression probability predictions with tailored feedback, including therapy resources for high-risk cases.

## Features

- **Exploratory Data Analysis (EDA):** Analyzes the dataset to identify key risk factors, revealing 58.5% depression prevalence and correlations with academic pressure (0.45) and financial stress (0.38).
- **Data Preprocessing:** Includes standardization of city names, encoding of categorical variables, scaling of numerical features, and feature engineering (e.g., degree groups, regional mapping).
- **Machine Learning Model:** Logistic Regression model optimized with GridSearchCV and a 0.45 decision threshold for high sensitivity in detecting at-risk students.
- **Web Application:** A Flask-based app with a responsive HTML interface, featuring a two-column form, dynamic gradient background, and a mental health-themed image. Outputs include probability predictions (e.g., "80% chance of depression") and therapy resource links for high-risk cases.
- **Therapy Support Page:** A dedicated page (`therapy.html`) with resources for users identified as high-risk.

## Dataset

The project uses the ["Student Depression Dataset"](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) from Kaggle, which includes features like:
- Age, CGPA, sleep duration, financial stress, academic pressure, suicidal thoughts, and more.
- Key findings: 58.5% of students reported depression, with suicidal thoughts and academic pressure as top predictors.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/student-depression-prediction.git
   cd student-depression-prediction
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset:**
   - Obtain the "Student Depression Dataset" from [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset).
   - Place the dataset file (e.g., `student_depression.csv`) in the `data/` folder.

5. **Run the Flask Application:**
   ```bash
   python app.py
   ```
   The app will be accessible at `http://localhost:5000`.

## Requirements

The `requirements.txt` file includes the following dependencies:
```
pandas
numpy
scikit-learn
flask
joblib
seaborn
matplotlib
```

You can generate the `requirements.txt` file using:
```bash
pip freeze > requirements.txt
```

## Usage

1. **Launch the Web Application:**
   - Run `python app.py` to start the Flask server.
   - Open `http://localhost:5000` in your browser.

2. **Input Student Data:**
   - Use the two-column form to enter student details (e.g., age, CGPA, academic pressure).
   - Submit the form to receive a depression probability prediction (e.g., "80% chance of depression").

3. **View Results:**
   - Low-risk cases display "You are healthy!" with encouraging feedback.
   - High-risk cases show a warning and a link to the therapy support page (`/therapy`) with resources.

4. **Explore Therapy Resources:**
   - For high-risk predictions, navigate to the therapy page for suggestions on therapy and support groups.

## Project Structure

```
student-depression-prediction/
├── data/
│   └── student depression dataset.csv    # Dataset file
├── static/
│   ├── images/                    # Mental health-themed images
│   └── styles.css                 # Tailwind CSS styles
├── templates/
│   ├── index.html                 # Main form interface
│   └── therapy.html               # Therapy resources page
├── models/
│   ├── depression_model.pkl       # Trained Logistic Regression model
│   └── scaler.pkl                 # StandardScaler for preprocessing
├── notebooks/
│   └── Main.ipynb                 # Jupyter Notebook for EDA and modeling
├── app.py                         # Flask application
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Model Performance

- **Algorithm:** Logistic Regression
- **Hyperparameters:** Optimized with GridSearchCV (C=1, L2 regularization)
- **Decision Threshold:** 0.45
- **Metrics:**
  - Accuracy: 0.86
  - Precision: 0.81
  - Recall: 0.9085
  - F1 Score: 0.8576
- **Key Predictors:** Suicidal thoughts (coefficient: 2.1), academic pressure (coefficient: 1.8)



## Future Improvements

- Incorporate additional features (e.g., social media activity, physical health metrics).
- Enable real-time data collection for dynamic predictions.
- Explore advanced algorithms like XGBoost or neural networks.
- Add user profiles, personalized recommendations, or institutional dashboards.
- Deploy on a cloud platform (e.g., AWS) for scalability.

## References

- Kaggle. (n.d.). *Student Depression Dataset*. [https://www.kaggle.com/datasets/hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Flask. (n.d.). *Flask documentation*. [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
- Tailwind CSS. (n.d.). *Tailwind CSS documentation*. [https://tailwindcss.com/docs](https://tailwindcss.com/docs)


## Contact

For questions or contributions, please contact [snehanshu.dev@gmail.com] or open an issue on GitHub.