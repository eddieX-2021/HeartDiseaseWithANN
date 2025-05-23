# Heart Disease Prediction App

This application predicts the risk of heart disease based on user inputs. It uses data preprocessing, scaling, and an Artificial Neural Network (ANN) model to output the probability of heart disease. The app is served using Streamlit, allowing for a simple and interactive web interface.

## What Does This App Do?

- **Data Preprocessing:**  
  The app uses a pickled preprocessor (a `ColumnTransformer`) and a scaler from scikit-learn that were fit on the training data. It processes features such as BMI, physical and mental health, demographic information, and behavioral aspects before feeding them into the model.

- **Model Prediction:**  
  The trained ANN model, built with TensorFlow/Keras, predicts the probability of heart disease based on the user provided inputs.

- **User Interface:**  
  The Streamlit web interface collects user input (like BMI, smoke habits, physical health, etc.) and displays the prediction result in an easily understandable format with outcomes like "High risk" or "Low risk" of heart disease.

## Tech Stack

- **Programming Language:**  
  Python

- **Libraries & Frameworks:**  
  - **Streamlit:** For building the web-based user interface.
  - **Pandas & NumPy:** For data manipulation and analysis.
  - **scikit-learn:** For data preprocessing (StandardScaler, OrdinalEncoder, OneHotEncoder) and model training support.
  - **TensorFlow / Keras:** For constructing, training, and deploying the ANN model.
  - **Pickle:** For serializing (saving and loading) the scaler and preprocessor.

## How to Run the App

1. **Clone the Repository:**  
   Clone this repository to your local machine.

2. **Install Dependencies:**  
   Create your own env
   conda or python activate it 
   Ensure you have Python installed (version 3.7 or later is recommended). Then install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**  
   Run the following command in your terminal from the project directory:
   ```bash
   streamlit run app.py
   ```
   This will launch the web app in your default browser.

## Project Structure

```
├── app.py                   # Streamlit app for user predictions
├── experiments.ipynb        # Notebook containing data preprocessing, model training, and pickling artifacts (preprocessor, scaler)
├── prediction.ipynb         # Notebook for demonstrating how to load the pickled artifacts and run predictions
├── heart_model.h5           # Trained ANN model (generated after training)
├── scaler.pkl               # Pickled scaler used for feature scaling
├── preprocessor.pkl         # Pickled ColumnTransformer for data preprocessing
└── README.md                # Project overview and instructions
```

## Model Training

The model is an Artificial Neural Network (ANN) built with TensorFlow/Keras. It was trained on preprocessed data from a heart disease dataset (`heart_2020_cleaned.csv`):

- **Preprocessing:**  
  The data is processed using a `ColumnTransformer` that applies:
  - Standard Scaling to numerical features (`BMI`, `PhysicalHealth`, `MentalHealth`, `SleepTime`)
  - One-hot encoding to nominal data (`Sex`, `Race`, `Diabetic`)
  - Ordinal encoding to ordered categorical data (`AgeCategory`, `GenHealth`)

- **Training:**  
  The data is split into training and test sets, then further scaled with a StandardScaler. The model architecture includes several dense layers with ReLU activations and a sigmoid output layer to produce a probability.

- **Saving Artifacts:**  
  The trained model (`heart_model.h5`), scaler (`scaler.pkl`), and preprocessor (`preprocessor.pkl`) are saved for later use in the prediction interfaces.

## Acknowledgments

- This project uses the heart disease dataset available from sources such as Kaggle.

Happy predicting!

try it https://heartdiseasewithann-f3jcwzuimgntrhpc6mg2uy.streamlit.app/  
