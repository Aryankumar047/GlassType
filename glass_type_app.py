import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix
# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df
glass_df = load_data()
# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]
# Creating the target series that holds last column.
y = glass_df['GlassType']
# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Full Dataset")
    st.dataframe(glass_df)
st.sidebar.subheader("Scatter Plot")
features_list = st.sidebar.multiselect("Select the x-axis values:",
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)
for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()
st.sidebar.subheader("Histogram")

# Choosing features for histograms.
hist_features = st.sidebar.multiselect("Select features to create histograms:",
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create histograms.
for feature in hist_features:
    st.subheader(f"Histogram for {feature}")
    plt.figure(figsize = (12, 6))
    plt.hist(glass_df[feature], bins = 'sturges', edgecolor = 'black')
    st.pyplot()
# Create box plots for all the columns.
# Sidebar for box plots.
st.sidebar.subheader("Box Plot")
# Choosing columns for box plots.
box_plot_cols = st.sidebar.multiselect("Select the columns to create box plots:",
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
# Create box plots.
for col in box_plot_cols:
    st.subheader(f"Box plot for {col}")
    plt.figure(figsize = (12, 2))
    sns.boxplot(glass_df[col])
    st.pyplot()
st.sidebar.subheader('Scatter Plot')
st.set_option('deprecation.showPyplotGlobalUse', False)
feats_list=st.sidebar.multiselect('Select X-axis values',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
for i in feats_list:
  plt.figure(figsize=(6,5))
  sns.scatterplot(glass_df[i],glass_df['Glasstype'])
  st.pyplot()
st.sidebar.subheader('Visulisation Selector')
plot_types=st.sidebar.multiselect('Select the charts of plots',('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))
if 'Histogram' in plot_types:
  st.subheader('Histogram')
  feature=st.sidebar.selectbox('Select the feature to create the histogram',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  plt.figure(figsize=(7,5))
  plt.hist(glass_df[feature],bins='sturges',edgecolor='black')
  st.pyplot()
if 'Boxplot' in plot_types:
  st.subheader('Boxplot')
  feature=st.sidebar.selectbox('Select the feature to create the boxplot',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  plt.figure(figsize=(7,5))
  sns.boxplot(glass_df[feature])
  st.pyplot()
if 'Count Plot' in plot_types:
  st.subheader('Count Plot')
  plt.figure(figsize=(7,5))
  sns.countplot(glass_df['Glasstype'])
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.
if 'Pie Chart' in plot_types:
  st.subheader('Pie chart')
  plt.figure(figsize=(7,5))
  count=glass_df['Glasstype'].value_counts()
  plt.pie(count,labels=count.index)
  st.pyplot()
# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
if 'Correlation Heatmap' in plot_types:
  st.subheader('Correlation Heatmap')
  #feature=st.sidebar.selectbox('Select the feature to create the histogram',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  plt.figure(figsize=(7,5))
  ax=sns.heatmap(glass_df.corr(),annot=True)
  bottom,top=ax.get_ylim()
  ax.set_ylim(bottom+0.5,top-0.5)
  st.pyplot()
# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function.
if 'Pair Plot' in plot_types:
  st.subheader('Pair Plot')
  plt.figure(figsize=(7,5))
  sns.pairplot(glass_df)
  st.pyplot()
st.slider.subheader('Select your values')
ri=st.sidebar.slider('Input RI',glass_df['Ri'].min(),glass_df['Ri'].max())
na=st.sidebar.slider('Input Na',glass_df['Na'].min(),glass_df['Na'].max())
mg=st.sidebar.slider('Input Mg',glass_df['Mg'].min(),glass_df['Mg'].max())
al=st.sidebar.slider('Input Al',glass_df['Al'].min(),glass_df['Al'].max())
si=st.sidebar.slider('Input Si',glass_df['Si'].min(),glass_df['Si'].max())
k=st.sidebar.slider('Input K',glass_df['K'].min(),glass_df['K'].max())
ca=st.sidebar.slider('Input Ca',glass_df['Ca'].min(),glass_df['Ca'].max())
ba=st.sidebar.slider('Input Ba',glass_df['Ba'].min(),glass_df['Ba'].max())
fe=st.sidebar.slider('Input Fe',glass_df['Fe'].min(),glass_df['Fe'].max())
st.sidebar.subheader('Choose Classifier')
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Random Forest Classifier','Logistic Regression'))
if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model Hyperparameters')
  c=st.sidebar.number_input('C',1,100,step=1)
  gamma=st.sidebar.number_input("Gama",1,100,step=1)
  kernel=st.sidebar.radio('Kernel',('Linear','RBF','Poly'))
  if st.sidebar.button('Classify'):
    st.sidebar.subheader('Support Vector Machine')
    svc_model=SVC(C=c,gamma=gamma,kernel=kernel)
    svc_model.fit(X_train,y_train)
    y_test_pred=svc_model.predict(X_test)
    accuracy=svc_model.score(X_test,y_test)
    glass_type=prediction(svc_model,ri,na,mg,al,si,k,ca,ba,fe)
    plot_confusion_matrix(svc_model,X_test,y_test)
    st.pyplot()
if classifier == 'Random Forest Classifier':
  st.sidebar.subheader('Model Hyperparameters:')
  n_estimators=st.sidebar.number_input('n_estimators',100,5000,step=1)
  max_depth=st.sidebar.number_input('Max Depth',1,20,step=1)
  if st.sidebar.button('Classify'):
    rfc_model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,n_jobs=-1)
    rfc_model.fit(X_train,y_train)
    rfc_model.score(X_test,y_test)
    glasstype=prediction(rfc_model,ri,na,mg,al,si,k,ca,ba,fe)
    plot_confusion_matrix(rfc_model,X_test,y_test)
    st.pyplot()
if classifier == 'Logistic Regression':
  st.sidebar.subheader("Model Hyperparameters:")
  c_val=st.sidebar.number_input('C',1,100,step=1)
  max_iter=st.sidebar.slider('Maximum iterations',100,500)
  if st.sidebar.button("Classify"):
    lr_model=LogisticRegression(c=c_val,max_iter=max_iter)
    lr_model.fit(X_train,y_train)
    lr_model.score(X_test,y_test)
    glasstype=prediction(lr_model,ri,na,mg,al,si,k,ca,ba,fe)
    st.write('Type of glass predicted is --> ',glasstype)
    accuracy=lr_model.score(X_test,y_test)
    st.write('Accuracy od the model --> ',accuracy)
    plot_confusion_matrix(lr_model,X_test,y_test)
    st.pyplot()
