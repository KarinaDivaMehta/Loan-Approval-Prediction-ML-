# Loan Status Prediction Model and API

This repository contains the complete workflow for building, evaluating, and deploying a **Classification Model** designed to predict the likelihood of a loan application being approved. The final model is wrapped in a **FastAPI** web service for easy integration into business applications.

## Project Overview

The core objective of this project is to leverage historical loan data to create an accurate predictive model. This model helps financial institutions automate and streamline the decision-making process for loan approvals, reducing manual effort and potential bias.

The process involves:

1.  **Data Preprocessing and Cleaning:** Handling missing values and preparing categorical features.
2.  **Feature Engineering and Scaling:** Transforming features to improve model performance.
3.  **Model Training and Evaluation:** Implementing and comparing **Logistic Regression** and **Support Vector Machine (SVM)** classifiers.
4.  **Hyperparameter Tuning:** Optimizing the best-performing model using **Randomized Search Cross-Validation**.
5.  **Deployment:** Creating a production-ready **RESTful API** using FastAPI to serve real-time predictions.

-----

## Table of Contents

  * [Installation](#installation)
  * [Data and Features](#data-and-features)
  * [Technical Workflow](#technical-workflow)
  * [API Usage](#api-usage)
  * [Model Performance](#model-performance)
  * [Technical Jargon Explained](#technical-jargon-explained)

-----

## Installation

To set up the project locally, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/KarinaDivaMehta/Loan-Approval-Prediction-ML-.git
    cd Loan-Approval-Prediction-ML-
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    The required packages are listed in the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the API:**
    Use `uvicorn` to start the FastAPI server.

    ```bash
    uvicorn loan_app:loan_app --reload
    ```

    The API will be available at `http://127.0.0.1:8000`. You can access the interactive documentation at `http://127.0.0.1:8000/docs`.

-----

## Data and Features

The model is trained on the `data/loan_data.csv` dataset, which includes the following features:

| Feature Name | Description | Data Type | Example Values |
| :--- | :--- | :--- | :--- |
| **Gender** | Applicant's gender | Categorical | Male, Female |
| **Married** | Marital status | Categorical | Yes, No |
| **Dependents** | Number of dependents | Categorical/Int | 0, 1, 2, 3+ |
| **Education** | Applicant's education level | Categorical | Graduate, Not Graduate |
| **Self\_Employed** | Is the applicant self-employed? | Categorical | Yes, No |
| **ApplicantIncome** | Applicant's monthly income | Numerical | 4000 |
| **CoapplicantIncome** | Co-applicant's monthly income | Numerical | 1500 |
| **LoanAmount** | Loan amount requested (in thousands) | Numerical | 120 |
| **Loan\_Amount\_Term** | Term of the loan (in months) | Numerical | 360 |
| **Credit\_History** | Applicant's credit history (1.0 - good, 0.0 - bad) | Numerical | 1.0, 0.0 |
| **Property\_Area** | Location of the property | Categorical | Urban, Semiurban, Rural |
| **Loan\_Status (Target)** | Loan approved or not | Categorical | Y (Yes), N (No) |

-----

## Technical Workflow

### 1\. Data Preprocessing

  * **Missing Value Imputation:** Missing values in 'Gender', 'Dependents', and 'Loan\_Amount\_Term' (columns with fewer missing values) were **dropped**. For columns with a higher number of missing values ('Self\_Employed', 'Credit\_History'), the **mode** was used for imputation.
  * **Feature Encoding:**
      * **Binary Columns** ('Gender', 'Married', 'Education', 'Self\_Employed', 'Loan\_Status') were mapped to 0 and 1.
      * **Ordinal/Categorical Columns** ('Property\_Area', 'Dependents') were converted using **One-Hot Encoding** to create numerical features, with the first category dropped to avoid **Multicollinearity**. The '3+' value in 'Dependents' was mapped to 3.

### 2\. Feature Scaling

Numerical features (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`) were scaled using **StandardScaler**. This standardization is crucial for distance-based algorithms like SVM and assists the convergence of Logistic Regression.

### 3\. Model Training and Selection

Two initial classification algorithms were compared using **Accuracy Score** and **Cross-Validation Score**:

  * **Logistic Regression**
  * **Support Vector Machine (SVM)**

### 4\. Hyperparameter Tuning

To maximize performance, the models were fine-tuned using **RandomizedSearchCV** with 5-fold cross-validation:

  * **Logistic Regression:** Optimized for the regularization parameter *C* and the *solver* (`liblinear`).
  * **SVM:** Optimized for the regularization parameter *C* and the *kernel* type (`linear`).

The **Tuned Logistic Regression Model** was selected as the final predictor due to its superior performance metrics.

### 5\. Final Model Persistence

The final, trained model object (`models/loan_status_predictor.pkl`) and the fitted **StandardScaler** object (`models/scaler.pkl`) were saved using the `joblib` library, making them ready for immediate deployment and inference.

-----

## API Usage

The project includes a FastAPI service that exposes a `/predict` endpoint for real-time inference.

### API Testing

Once the API server is running, you can test the prediction endpoint using two methods: the built-in interactive documentation (Swagger UI) or an external client like Postman.

#### Using Interactive Documentation (Swagger UI)

This is the simplest way to confirm your API is working correctly.

  * **Access the Docs:** Open your web browser and navigate to the documentation URL: `http://127.0.0.1:8000/docs`
  * **Locate Endpoint:** Find the **`POST /predict`** endpoint.
  * **Test:** Click the **"Try it out"** button, modify the example JSON body if desired, and click **"Execute"**. The response will show the HTTP status code and the model's prediction (e.g., `{"Loan Status": "Approved"}`).

#### Prediction Endpoint

  * **Endpoint:** `/predict`
  * **Method:** `POST`
  * **Input Data:** A JSON object matching the required schema.

| Key | Description | Type | Example |
| :--- | :--- | :--- | :--- |
| `Gender` | 0=Male, 1=Female | float | 1.0 |
| `Married` | 0=No, 1=Yes | float | 1.0 |
| `Education` | 0=Not Graduate, 1=Graduate | float | 1.0 |
| `Self\_Employed` | 0=No, 1=Yes | float | 0.0 |
| `ApplicantIncome` | Monthly income | float | 5000.0 |
| `CoapplicantIncome` | Co-applicant's monthly income | float | 1500.0 |
| `LoanAmount` | Requested loan amount | float | 120.0 |
| `Loan\_Amount\_Term` | Term in months | float | 360.0 |
| `Credit\_History` | 0.0 or 1.0 | float | 1.0 |
| `Property\_Area` | 0=Rural, 1=Semiurban, 2=Urban | float | 1.0 |

**Example Request Body (JSON):**

```json
{
    "Gender": 1.0,
    "Married": 1.0,
    "Education": 1.0,
    "Self_Employed": 0.0,
    "ApplicantIncome": 5000.0,
    "CoapplicantIncome": 1500.0,
    "LoanAmount": 120.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": 1.0,
    "Dependents": 2.0
}
```

**Example Response:**

```json
{
    "Loan Status": "Approved"
}
```

-----

## Model Performance

The model evaluation process compared several classification algorithms. After initial testing, the Logistic Regression and Support Vector Classifier (SVC) were selected for hyperparameter tuning.

The final, tuned models demonstrated the following performance profile, with the best score achieved through **cross-validation** during tuning:

| Model | Average Cross-Validation Score (Pre-Tuning) | Best Score (Post-Tuning) |
| :--- | :--- | :--- |
| **RandomForestClassifier** | $\sim 0.83$ | - |
| **SVC** | $\sim 0.84$ | $\sim 0.84$ |
| **DecisionTreeClassifier** | $\sim 0.77$ | - |
| **LogisticRegression** | $\sim 0.84$ | $\sim 0.84$ |
| **GradientBoostingClassifier** | $\sim 0.81$ | - |

Both the Tuned **Logistic Regression** and **SVC** models achieved the highest score of $\sim 0.84$. The Logistic Regression model was ultimately chosen for deployment.
 
-----

## Technical Jargon Explained

  * **Classification Model:** A supervised machine learning model that predicts a **categorical** output (e.g., Yes/No, Approved/Not Approved).
  * **Logistic Regression:** A linear model used for classification tasks. It calculates the probability of a binary event occurring.
  * **Support Vector Machine (SVM):** A non-linear classification model that finds the optimal **hyperplane** (a decision boundary) to separate different classes in the feature space.
  * **One-Hot Encoding:** A process of converting nominal categorical features into a set of binary (0 or 1) columns that machine learning models can process.
  * **Multicollinearity:** A statistical phenomenon where two or more predictor variables in a model are highly correlated, which can lead to unstable and misleading model coefficients.
  * **StandardScaler:** A data preprocessing technique that **standardizes** features by removing the mean and scaling to unit variance. This prevents features with larger magnitudes from unfairly dominating the model training process.
  * **Cross-Validation:** A model evaluation technique where the data is split into multiple subsets (folds). The model is trained and tested iteratively on different folds to ensure its performance metric is robust and generalized.
  * **Hyperparameter Tuning:** The process of optimizing the non-trainable settings of an algorithm (like the regularization strength *C*) to improve model performance on unseen data.
  * **FastAPI:** A modern, high-performance web framework for building APIs with Python, known for its speed and automatic documentation generation.
  * **RESTful API:** An application programming interface (API) that uses standard HTTP methods (like POST) to allow clients to interact with data and resources on a server.
