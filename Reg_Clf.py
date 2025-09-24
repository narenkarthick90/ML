# import streamlit as st
# import pandas as pd
# import numpy as np

# st.title("Would you buy a house in 2025?")
# st.write("This is a regression and classification model to predict house prices and whether you would buy a house in 2025.")

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import mean_squared_error, accuracy_score
# loan = pd.read_csv("loan.csv")
# housing = pd.read_csv("housing.csv")

# # Select required columns
# loan_features = loan[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']]
# housing_features = housing[['Location', 'Area', 'Bedrooms', 'Price']]

# # Encode categorical (loan status, location)
# loan_features['Loan_Status'] = LabelEncoder().fit_transform(loan_features['Loan_Status'])
# housing_features['Location'] = LabelEncoder().fit_transform(housing_features['Location'])

# # Simulate merge: both have same number of records (or sample)
# combined = pd.concat([loan_features.reset_index(drop=True), housing_features.reset_index(drop=True)], axis=1)


# X_reg = combined[['Location', 'Area', 'Bedrooms', 'ApplicantIncome', 'CoapplicantIncome']]
# y_reg = combined['Price']

# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
# reg_model = LinearRegression()
# reg_model.fit(X_train_reg, y_train_reg)

# y_pred_reg = reg_model.predict(X_test_reg)
# print("RMSE (Price Prediction):", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))


# X_cls = combined[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
# y_cls = combined['Loan_Status']

# X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
# cls_model = LogisticRegression(max_iter=1000)
# cls_model.fit(X_train_cls, y_train_cls)

# y_pred_cls = cls_model.predict(X_test_cls)
# print("Accuracy (Loan Approval):", accuracy_score(y_test_cls, y_pred_cls))

# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

@st.cache(allow_output_mutation=True)
def load_data():
    house = pd.read_csv("housing.csv")  # from Kaggle
    loan = pd.read_csv("loan.csv")     # from Kaggle
    return house, loan

house_df, loan_df = load_data()

st.title("🏠 House Price & Loan Approval Predictor")

st.sidebar.header("Select Task")
task = st.sidebar.selectbox("Choose:", ["House Price Regression", "Loan Approval Classification"])

if task == "House Price Regression":
    st.header("Predict House Price")

    df = house_df.dropna()
    X = df[['location', 'area', 'bedrooms']]
    y = df['price']

    X_enc = pd.get_dummies(X, columns=['location'])
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.write("### Model Performance:")
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    st.write(f"RMSE: {rmse:.2f}")

    st.write("### Predict for your house:")
    loc = st.selectbox("Location", options=df['location'].unique())
    area = st.number_input("Area (sq ft)", value=1000)
    beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    if st.button("Predict House Price"):
        inp = pd.DataFrame([[area, beds]], columns=['area','bedrooms'])
        for l in X_enc.columns:
            if l.startswith('location_'):
                inp[l] = 1 if l == f"location_{loc}" else 0
        price = model.predict(inp)[0]
        st.success(f"Estimated Price: ₹{price:,.0f}")

else:
    st.header("Predict Loan Approval")

    df = loan_df.dropna()
    df['Loan_Status'] = LabelEncoder().fit_transform(df['Loan_Status'])
    X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']]
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    st.write("### Model Performance:")
    acc = accuracy_score(y_test, model.predict(X_test))
    st.write(f"Accuracy: {acc*100:.2f}%")

    st.write("### Check your loan eligibility:")
    ai = st.number_input("Applicant Income", value=5000)
    ci = st.number_input("Coapplicant Income", value=0)
    amt = st.number_input("Loan Amount", value=100)
    ch = st.selectbox("Credit History", options=[0.0, 1.0])
    if st.button("Check"):
        val = np.array([[ai, ci, amt, ch]])
        out = model.predict(val)[0]
        st.success("✅ Loan Approved" if out == 1 else "❌ Loan Not Approved")
