## link : https://cmse-mid-term.herokuapp.com/

import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.figure_factory as pl
import matplotlib.pyplot as plt
import hiplot as hip
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

st.write(""" 
# Healthcare Institutions Situation Analysis to Improve Services
""")

df = pd.read_csv('before_outlier_new.csv')

df = df.sort_values(['Age'])

# st.write(df.columns.values)
column_name = df.index.tolist()
columns = ['case_id','Hospital_code','Hospital_type_code','City_Code_Hospital','Hospital_region_code','Available Extra Rooms in Hospital','Department','Ward_Type','Ward_Facility_Code','Bed Grade','patientid','City_Code_Patient','Type of Admission','Severity of Illness','Visitors with Patient','Age','Admission_Deposit','Stay']
category_columns = ['Hospital_code','Hospital_type_code','City_Code_Hospital','Hospital_region_code','Available Extra Rooms in Hospital','Department','Ward_Type','Ward_Facility_Code','Bed Grade','patientid','City_Code_Patient','Type of Admission','Severity of Illness','Visitors with Patient','Age','Stay']
numerical_colunms = ['case_id','Admission_Deposit']
st.sidebar.write("# 😎Select appropriate option which you want")
ratio = st.sidebar.radio("",('❓Problem Statements & 📅Data','📊Data Analysis','🧑‍💻Model Development','🤔💭Bussiness Conclusion'))


x_train= df.drop(['Stay'],axis=1)
y_train = df['Stay']
### validation accuracy by cross validation
def average_validation(model):
    x_train= x_train
    y_train = y_train
    scores = cross_val_score(model,x_train,y_train,cv=5).mean()
    return scores

if(ratio == '❓Problem Statements & 📅Data'):
    st.write("""
    👉 Nowadays, health is becoming the predominant part of any individual's life. During the COVID-19 outbreak, healthcare industries play a vital role to break the chain of infection. But in that situation, some hospitals and healthcare institutes have to suffer from heavy patient traffic and were not able to mitigate every patient's demands because of unavailability of beds. While healthcare management has various use cases for using data science, patient length of stay is one critical parameter to observe and predict if one wants to improve the efficiency of the healthcare management in a hospital. This perameters will help to identify the expected stay duration for any new patient and that according to that the hospital staff can optimize the treatment plan in such a way that the expected stay time can be reduced so that the risk of getting an infection from the patient for health worker can be reduced.

    👉 This parameter helps hospitals to identify patients of high LOS risk (patients who will stay longer) at the time of admission. Once identified, patients with high LOS risk can have their treatment plan optimized to miminize LOS and lower the chance of staff/visitor infection. Also, prior knowledge of LOS can aid in logistics such as room and bed allocation planning.
    """)

    st.write("## Problem Statement")
    st.write("## Data Description")
    st.write("""
            ###### ➡️ case_id : Case_ID registered in Hospital
            ###### ➡️ Hospital_code : Unique code for the Hospital
            ###### ➡️ Hospital_type_code : Unique code for the type of Hospital
            ###### ➡️ City_Code_Hospital : City Code of the Hospital
            ###### ➡️ Hospital_region_code : Region Code of the Hospital
            ###### ➡️ Available Extra Rooms in Hospital : Number of Extra rooms available in the Hospital
            ###### ➡️ Department : Department overlooking the case
            ###### ➡️ Ward_Type : Code for the Ward type
            ###### ➡️ Ward_Facility_Code : Code for the Ward Facility
            ###### ➡️ Bed Grade : Condition of Bed in the Ward
            ###### ➡️ patientid : Unique Patient Id
            ###### ➡️ City_Code_Patient : City Code for the patient
            ###### ➡️ Type of Admission : Admission Type registered by the Hospital
            ###### ➡️ Severity of Illness : Severity of the illness recorded at the time of admission
            ###### ➡️ Visitors with Patient : Number of Visitors with the patient
            ###### ➡️ Age : Age of the patient
            ###### ➡️ Admission_Deposit : Deposit at the Admission Time
            ###### ➡️ Stay : Stay Days by the patient
        """)

    st.write(" ")
    st.write("""


    👉 Here "Stay" coloum is target variable , which we want to predict at the time of new patient comes

    👉 In primary EDA, I have found 2 columns (Bed Grade, City_Code_Patient) having null values about(<1 % , 1.4%) for each , And the missingness is MNCR type:

    👉 From primary analysis, I have found that the dataset does not have the null values more than 2% so that It seems purposful in tems of my problem statement.  I am planning to provide EDA for finding trends and than apply some ML models such as linear regression or any higher in-order to build a predictive model which predicts the expected time of stay for new patient based on patient symptoms. In the primary studies, I have performed EDA tasks such as finding statistical values such as mean of the data, Check for missing data and finding the percentage of the data, Also map that missing values for finding the missingness of the data. Till now I haven't worked upon the outlier detection as well as encoding part 

    👉 I am thinking of making the webapp in such way that the first section of the application will display the statistics of the data such as graph and corelation with the target variable fo different field. Then I am thinking to add some enable diable button throught which we can change the data such that the model use standardize data to predict or not . At last want to provide graph for the model accuracy and field in which person enter data and than based on that the model predict the expected number of stay. Through this the healthcare institute can get idea of stay for perticular patient.

    👉 In order to make my project work more interective and easy to understand which speak out the story of the data, I am thinking if using plotly as well as altplot because the are interective so that it will be helpful in detail observation. Along with that I will use hiplot to visulalise the overall data clustering on the bases of various categorical columns. The parellel plot from pandas also helpful during feature engineering phase because in that we do lot of normalization as well as outliers removal so we can compare both the situation parellally. Along with all, I will use matplotlib for quick analysis.

    👉 From all of the above plot, I will mention necessary plot on my streamlit web-app so that its good  for reader to get story 

    👉 From my perspective, in this highly infected time, any disease can spread very easily and it can affect a community. So for maintaining the proper healthcare system, Proper infrastructure requires to meet the patient demand in a crucial time and for that I think this kind of predictive system might be helpful for management team. 
    """)


if(ratio == '📊Data Analysis'):
    # st.image('data analysis.jpg',width=500)
    st.title("Data Analysis")
    
    st.write("### Numerical Columns")
    st.write("""
            ###### ➡️ Admission_Deposit
            ###### ➡️ Available Extra Rooms in Hospital""")

    st.write("")
    st.write("### Categorical Columns")
    st.markdown("""      
            ###### ➡️ Hospital_type_code  
            ###### ➡️ City_Code_Hospital 
            ###### ➡️ Hospital_region_code	
            ###### ➡️ Department	
            ###### ➡️ Ward_Type	Ward_Facility_Code	
            ###### ➡️ Bed Grade	patientid	 
            ###### ➡️ City_Code_Patient	
            ###### ➡️ Type of Admission	
            ###### ➡️ Severity of Illness	
            ###### ➡️ Visitors with Patient	
            ###### ➡️ Age
            ###### ➡️ Stay""")

    st.subheader("👇Analyse the data by graph👇")
    st.write("")
    graph = st.selectbox("Select Chart type",["Pairplot","Violin Plot","Box Plot","Hi Plot","Dist Plot","Joint Plot","Lm Plot"])

    if graph == "Hi Plot":
        st.write("")
        st.subheader("Parameters:")
        options = st.multiselect('Select Feature for hiplot',columns,[])
        if len(options) > 0:       
            temp_data = df.loc[:100,options].reset_index()

            st.write(temp_data.head())
            xp = hip.Experiment.from_dataframe(temp_data[1:])
            xp.to_streamlit(ret="selected_uids", key="hip").display()

    if graph == "Pairplot":
        st.write("")
        st.subheader("Parameters:")
        pair_options = st.multiselect('Select Feature for pairplot',columns,[])
        pair_hue = st.selectbox('Select hue for pairplot',category_columns)
        if len(pair_options) > 0 :       
            temp_data = df.loc[:100,pair_options].reset_index()
            temp_data[pair_hue] = df.loc[:,pair_hue]
            st.write(temp_data.head())
            pairplot = sns.pairplot(temp_data,hue=pair_hue)
            st.pyplot(pairplot)


    if graph == "Violin Plot":
        st.write("")
        st.subheader("Parameters:")
        violin_x = st.selectbox('X-axis for Violin plot',columns,index=columns.index('Severity of Illness'))
        violin_y = st.selectbox('Y-axis for Violin plot',category_columns,index=category_columns.index('City_Code_Hospital'))
        vio_plot = sns.catplot(x=violin_x, y=violin_y, kind="violin", aspect=2,data=df[:1000])
        plt.xlabel(violin_x, size=14)
        plt.ylabel(violin_y, size=14)
        plt.title("Violin Plot with Seaborn Catplot", size=20)
        # arr = np.random.normal(1, 1, size=100)
        # fig, ax = plt.subplots()
        # ax.violinplot()
        st.pyplot(vio_plot)

    if graph == "Box Plot":
        st.write("")
        st.subheader("Parameter:")
        fig = plt.figure(figsize=(12, 6))
        box_x = st.selectbox('X-axis for Box plot',columns,index=columns.index('Severity of Illness'))
        box_y = st.selectbox('Y-axis for Box plot',category_columns,index=category_columns.index('City_Code_Hospital'))
        box_plot = sns.catplot(x=box_x, y=box_y, kind="box", aspect=2,data=df[:1000])
        plt.xlabel(box_x, size=14)
        plt.ylabel(box_y, size=14)
        plt.title("Box Plot", size=20)
        st.pyplot(box_plot)

    if graph == "Dist Plot":
        st.write("")
        st.subheader("Parameter:")
        dist_x = st.selectbox('X-axis for Box plot',columns,index=columns.index('Severity of Illness'))
        dist_y = st.selectbox('Y-axis for Box plot',columns,index=columns.index('City_Code_Hospital'))
        dist_hue = st.selectbox('Hue for Box plot',category_columns,index=category_columns.index('Department'))
        dist_plot = sns.jointplot(data=df[:100], x=dist_x, y=dist_y,hue= dist_hue,kind="kde")
        st.pyplot(dist_plot)
    
    if graph == "Joint Plot":
        st.write("")
        st.subheader("Parameter:")
        joint_x = st.selectbox('X-axis for Box plot',columns,index=columns.index('Severity of Illness'))
        joint_y = st.selectbox('Y-axis for Box plot',columns,index=columns.index('City_Code_Hospital'))
        joint_hue = st.selectbox('Hue for Box plot',category_columns,index=category_columns.index('Department'))
        joint_plot = sns.jointplot(data=df[:100], x=joint_x, y=joint_y,hue= joint_hue)
        st.pyplot(joint_plot)

    if graph =="Lm Plot":
        st.write("")
        st.subheader("Parameter:")
        lm_x = st.selectbox('X-axis for Box plot',columns,index=columns.index('Severity of Illness'))
        lm_y = st.selectbox('Y-axis for Box plot',columns,index=columns.index('Severity of Illness'))
        lm_hue = st.selectbox('Hue for Box plot',category_columns,index=category_columns.index('Department'))
        lm_plot = sns.lmplot(data=df, x=lm_x, y=lm_y,hue= lm_hue)
        st.pyplot(lm_plot)

data = pd.read_csv("train.csv")
from sklearn.model_selection import train_test_split


train_x,test_x,train_y,test_y = train_test_split(df.drop(['Stay'],axis=1), df.Stay, test_size=0.33 ,random_state=42)
train=pd.concat([train_x,train_y],axis=1)
test = test_x
train = train.drop(['Hospital_region_code', 'Bed Grade', 'patientid', 'City_Code_Patient'], axis = 1)
test = test.drop(['Hospital_region_code', 'Bed Grade', 'patientid', 'City_Code_Patient'], axis = 1)
X_train = train.drop([ 'Stay'], axis=1)
Y_train = train["Stay"]
X_test  = test
columns_list = x_train.columns[2:]

stay_dict = {0:'0-10', 1:'11-20', 2:'21-30', 3:'31-40', 4:'41-50', 5:'51-60', 5:'61-70',7: '71-80', 8:'81-90', 9: '91-100', 10:'More than 100 Days'}
import pickle
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
knn_model = pickle.load(open('knn_model', 'rb'))
# x_test = pd.read_csv("C:\\Users\\kevin\\OneDrive\\Documents\\MSU\\1st Sem\\CMSE 830  introduction to data science\\Mid Project\\Intro_to_datascience-_CMSE\\split_final_test_x.csv")  
# y_test = pd.read_csv("C:\\Users\\kevin\\OneDrive\\Documents\\MSU\\1st Sem\\CMSE 830  introduction to data science\\Mid Project\\Intro_to_datascience-_CMSE\\split_final_test_y.csv") 


if(ratio == '🧑‍💻Model Development'):
    # st.write(columns)
    st.title("Model Development")
    st.write("Still under Development👨‍💻👨‍💻")
    # type = st.radio(
    #     "Select type of algorithm",
    #     ("Supervised Algorithm","Unsupervised Algorithm")
    # )
    
    st.title("Supervised Algorithm")
    supervised_algo = st.radio(
        "Select algorithm",
        ( "k-nearest neighbour","Decision Tree Classification","Stochastic Gradient Descent","Logistic Classifier")
    )
    # if(supervised_algo == "Support-Vector-Machine (SVM)"):
    #         st.write("Support-Vector-Machine (SVM)")
    #         # svm_model = svm.SVC(decision_function_shape='ovo')
    #         # svm_model.fit(x_train, y_train)

    #         # scores = cross_val_score(svm_model)


    if(supervised_algo=="k-nearest neighbour"):
            st.write("K-nearest Neighbour")
            # knn_prediction = knn_model.predict(x_test)
            para = st.multiselect('Select Feature for decision tree',columns_list,[])
            n_neighbour = st.slider('N_neighbour?', 2, 100, 3)
            if(st.button('Generate Model')):
                knn = KNeighborsClassifier(n_neighbors = n_neighbour)
                knn.fit(X_train.loc[:,para], Y_train)
                Y_pred = knn.predict(X_test.loc[:,para])
                acc_knn = round(knn.score(X_train.loc[:,para], Y_train) * 100, 2)
                st.write("Accuracy =",acc_knn)
                st.subheader("Classification Report")
                #st.write(classification_report(Y_train,knn.predict(X_train.loc[:,para])))
                st.write(pd.DataFrame(classification_report(Y_train,knn.predict(X_train.loc[:,para]),output_dict=True)).transpose())
                st.subheader("Cconfusion Matrixs")
                fig4 = plt.figure()
                sns.heatmap(confusion_matrix(Y_train,knn.predict(X_train.loc[:,para])))
                st.pyplot(fig4)
            st.title("Predict Staylength")
            pred_para= []
            st.write("Age data encoding")
            st.write("{'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9}")
            for i in para:
                temp = st.selectbox(f"select value of {i} to predict",x_train[i].unique())
                pred_para.append(temp)
                # st.write(pred_para)
                
                
            if (st.button("Predict")):
                knn = KNeighborsClassifier(n_neighbors = n_neighbour)
                knn.fit(X_train.loc[:,para], Y_train)
                ans = knn.predict([pred_para] )
                print(ans)
                st.write("it expected that the person will stay for ",stay_dict[ans[0]], "Days.")


    elif(supervised_algo=="Stochastic Gradient Descent"):
            para = st.multiselect('Select Feature for decision tree',columns_list,[])
            iter = st.slider('Max Iteration', 2, 100, 5)
            if(st.button('Generate Model')):
                SCG_classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=iter)
                SCG_classifier.fit(X_train.loc[:,para], Y_train)
                Y_pred = SCG_classifier.predict(X_test.loc[:,para])
                acc_SCG_classifier = round(SCG_classifier.score(X_train.loc[:,para], Y_train) * 100, 2)
                st.write("Accuracy = ",acc_SCG_classifier)
                st.subheader("Classification Report")
                # st.write(classification_report(Y_train,SCG_classifier.predict(X_train.loc[:,para])))
                st.write(pd.DataFrame(classification_report(Y_train,SCG_classifier.predict(X_train.loc[:,para]),output_dict=True)).transpose())
                st.subheader("Cconfusion Matrixs")
                fig4 = plt.figure()
                sns.heatmap(confusion_matrix(Y_train,SCG_classifier.predict(X_train.loc[:,para])))
                st.pyplot(fig4)
            # test_acc_SCG_classifier = round(SCG_classifier.score(X_test.iloc[:,[1,2,3,7,8]], test_y) * 100, 2)
            st.title("Predict Staylength")
            st.write("Age data encoding")
            st.write("{'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9}")
            pred_para= []
            for i in para:
                temp = st.selectbox(f"select value of {i} to predict",x_train[i].unique())
                pred_para.append(temp)
                # st.write(pred_para)   
                
            if (st.button("Predict")):
                SCG_classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=iter)
                SCG_classifier.fit(X_train.loc[:,para], Y_train)
                ans = SCG_classifier.predict([pred_para] )
                print(ans)
                st.write("it expected that the person will stay for ",stay_dict[ans[0]], "Days.")

    elif(supervised_algo=="Logistic Classifier"):
            para = st.multiselect('Select Feature for decision tree',columns_list,[])
            # iter = st.slider('Max Iteration', 2, 100, 5)
            if(st.button('Generate Model')):
                logi_classifier =  linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
                logi_classifier.fit(X_train.loc[:,para], Y_train)
                Y_pred = logi_classifier.predict(X_test.loc[:,para])
                acc_logi_classifier = round(logi_classifier.score(X_train.loc[:,para], Y_train) * 100, 2)
                st.write("Accuracy =",acc_logi_classifier)
                st.subheader("Classification Report")
                # st.write(classification_report(Y_train,logi_classifier.predict(X_train.loc[:,para])))
                st.write(pd.DataFrame(classification_report(Y_train,logi_classifier.predict(X_train.loc[:,para]),output_dict=True)).transpose())
                st.subheader("Cconfusion Matrixs")
                fig4 = plt.figure()
                sns.heatmap(confusion_matrix(Y_train,logi_classifier.predict(X_train.loc[:,para])))
                st.pyplot(fig4)
            st.title("Predict Staylength")
            st.write("Age data encoding")
            st.write("{'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9}")
            pred_para= []
            for i in para:
                temp = st.selectbox(f"select value of {i} to predict",x_train[i].unique())
                pred_para.append(temp)
                # st.write(pred_para)
                
                
            if (st.button("Predict")):
                logi_classifier =  linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
                logi_classifier.fit(X_train.loc[:,para], Y_train)
                ans = logi_classifier.predict([pred_para] )
                print(ans)
                st.write("it expected that the person will stay for ",stay_dict[ans[0]], "Days.")
    else:
            st.write("Decision Tree Classification")
            decision_tree = DecisionTreeClassifier()
            para = st.multiselect('Select Feature for decision tree',columns_list,[])
            if(st.button('Generate Model Report')):
                decision_tree.fit(X_train.loc[:,para], Y_train)
                #Y_pred = decision_tree.predict(X_test.loc[:,[1,2,3,4,5,7,8]])
                acc_decision_tree = round(decision_tree.score(X_train.loc[:,para], Y_train) * 100, 2)
                st.write("Accuracy = " , acc_decision_tree)
                st.subheader("Classification Report")
                st.write(pd.DataFrame(classification_report(Y_train,decision_tree.predict(X_train.loc[:,para]),output_dict=True)).transpose())
                st.subheader("Cconfusion Matrixs")
                fig4 = plt.figure()
                sns.heatmap(confusion_matrix(Y_train,decision_tree.predict(X_train.loc[:,para])))
                st.pyplot(fig4)
            st.title("Predict Staylength")
            st.write("Age data encoding")
            st.write("{'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9}")
            pred_para= []
            for i in para:
                temp = st.selectbox(f"select value of {i} to predict",x_train[i].unique())
                pred_para.append(temp)
                # st.write(pred_para)
                
                
            if (st.button("Predict")):
                decision_tree.fit(X_train.loc[:,para], Y_train)
                ans = decision_tree.predict([pred_para] )
                print(ans)
                st.write("it expected that the person will stay for ",stay_dict[ans[0]], "Days.")



    
    # hospital_code = st.selectbox('Select hospital code',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
    # hospital_type = st.selectbox('Select hospital type',['a','b','c','d','e','f','g'])

    # city_code_patient = st.selectbox('Select city code', [ 7.,  8.,  2.,  5.,  6.,  3.,  4.,  1.,  9., 14., 25., 15., 12., 10., 28., 24., 23., 20., 11., 13., 21., 18., 16., 26., 27., 22., 19., 31., 34., 32., 30., 29., 37., 33., 35.,36., 38.])
    # hospital_region_code =  st.selectbox('Select region code', [1,2,3])
    # available_room = st.selectbox('Select Available Room',[ 3,  2,  1,  4,  6,  5,  7,  8,  9, 10, 12,  0, 11, 20, 14, 21, 13, 24])
    # department = st.selectbox('select department',['radiotherapy', 'anesthesia', 'gynecology', 'TB & Chest disease', 'surgery'])
    # Severity_of_Illness = st.selectbox('Severity of Illness',[0,1,2])
    # Type_of_Admission = st.selectbox("Type of Admission",[0,1,2])
    # Age = st.selectbox("Age",[1,2,3,4,5,6,7,8,9,10])
    # department_val ={
    #   'radiotherapy':1, 'anesthesia':2, 'gynecology':3, 'TB & Chest disease':4, 'surgery':5
    # }
    # data = [available_room, department_val[department],Type_of_Admission,Severity_of_Illness,Age]

    # loaded_model = pickle.load(open("knn_model", 'rb'))
    # print(loaded_model.predict(data))


if(ratio == '🤔💭Bussiness Conclusion'):
    st.title("Business Stories")
    st.write("# Age wise analysis")
    # df['Bed Grade']=df['Bed Grade'].fillna(df.mean()['Bed Grade'])

    
    age_unique = df['Age'].unique()
    st.sidebar.write('Age wise analysis')
    selected_age = st.sidebar.selectbox('Select age range',age_unique)

    #histogram for age
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['Age'])
    st.pyplot(fig,size=(1,1))
    st.write("""
        👉 From the overall distribution, we find that most of the patient of the instutute are 3 & 4 age-group, which represent age 31-40 ahd 41-50 respectively.

        👉 In ferther analysis, we only consider those groups of the patient because they are in high proportion.""")


    age_df = df[df['Age'] == selected_age]

    st.write("""#### To resolve the issue of waiting in hospitals, we are analysing severity of illness as well as departments of cases""")
    ## for checking severity of admissions and types of admission with age
    severity_data = age_df.groupby(['Severity of Illness']).count()['Age'].values
    severity_labels = age_df.groupby(['Severity of Illness']).count()['Age'].index.values
    colors = sns.color_palette('pastel')[0:3]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(severity_data, labels = severity_labels, colors = colors, autopct='%.02f%%')
    ax.set_title(f"Percentage Severity of Illness for {selected_age} age group")
    st.pyplot(fig,size = (1,1))
    fig.ax = plt.subplot()

    st.write("""
    👉 From above graph, we can say that for each age groups moderate severity cases are comparitely dominant to extream and minor cases.

    👉 espacialy for age group 30-40 and 40-50 , moderate severisty of illness is almost 3/5th of overall cases.

    👉 Thus institued need to focus on those kind of cases more.So that most of the patient who having moderate level illness, can get treeeted as soon as possible so their length of stay can be shorten and hospital's infrasterucutre can become available for others . It will help to reduce the panic situation like the situation of COVID-19 which we face couple of years back when the beds for critical patient are also not available.
    """)

    ## for department of cases
    department_data = age_df.groupby(['Department']).count()['Age'].values
    print(department_data)
    department_labels = age_df.groupby(['Department']).count()['Age'].index.values
    colors = sns.color_palette('pastel')[0:5]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(department_data, labels = department_labels, colors = colors, autopct='%.0f%%')
    ax.set_title(f"Percentage for each department for {selected_age} age group")
    st.pyplot(fig,size = (1,1))
    fig.ax = plt.subplot()
    st.write("""
    #### 👉 Departmental analysis : 
    👉 Throught this, I found that most of the cases for all the age groups are facing more gynecology related issues. So improvement of gynecology department will be benifitial.
    
    👉 Also most of the cased are associate with gynacology department. So there might be probability of mass infection. So survey need to take amongs the patient to go tho depth of infection.  
    """)


    st.write("# City wise analysis")
    ### for city for hospital
    city_hospital_unique = df['City_Code_Hospital'].unique()

    selected_city = st.sidebar.selectbox('Select city for getting hospitle data',city_hospital_unique)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['City_Code_Hospital'])
    st.pyplot(fig,size=(1,1))

    city_data = df[df['City_Code_Hospital'] == selected_city]
    fig = plt.figure(figsize=(10, 4))
    st.write(f"### Age wise distribution for '{selected_city}' city")
    sns.countplot(x = "Age", data = city_data)
    st.pyplot(fig)

    severity_data = city_data.groupby(['Severity of Illness']).count()['Age'].values
    severity_labels = age_df.groupby(['Severity of Illness']).count()['Age'].index.values
    colors = sns.color_palette('pastel')[0:3]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(severity_data, labels = severity_labels, colors = colors, autopct='%.02f%%')
    ax.set_title(f"Percentage Severity of Illness for {selected_city} city")
    st.pyplot(fig,size = (1,1))
    fig.ax = plt.subplot()

    st.write("""
        👉 From above distribution chart, We can see most of the cased are belong to city '2'. This is because of 2 reason:
            
            1st : May having not good Hostpitalization facility
            2nd : Mass infection for perticular city

        👉 Both the situation can be tackled by improving infrastructure for perticular City.

        👉 Also, We have find similar trend for all the city for age groups and severity of illness. But for some city having specifically older age group patient who having issue which is extream. So those patient having high Loss of risk percentage. So those can't be ignored to provide better infrastructure
        """)



    # for hospital in city_hospital_unique:
    #     print( hospital )
    #     print(df[df['City_Code_Hospital'] == hospital]['City_Code_Patient'].unique())

    # fig, ax = plt.subplots()
    # df_dept=age_df.groupby('Department')['case_id'].count()

    # print(df_dept.index.values)

    # fig = plt.figure(figsize=(10, 4))
    #     sns.countplot(x = "year", data = data_frame)
    #     st.pyplot(fig)


    selected_age_city = st.sidebar.select_slider(label="select_age",options=age_unique)
    selected_severity_city = st.sidebar.select_slider(label="select severity",options= severity_labels,value=0)


    amount_df = df[df['Age'] == selected_age_city ]
    # amount_df = amount_df[amount_df['Severity of Illness'] == selected_severity_city]
    #st.write(amount_df)
    fig= fig1 = plt.figure(figsize=(20, 41))
    st.write(f"# Data analysis for '{selected_age_city}' age range")
    fig = sns.catplot(data=amount_df, x="Type of Admission", y="Admission_Deposit",col="Severity of Illness" ,kind="point")
    st.pyplot(fig)

    fig1 = sns.catplot(
        data=amount_df, x="Type of Admission", y="Visitors with Patient", col='Severity of Illness',
        capsize=.2, palette="YlGnBu_d", errorbar="se",
        kind="point", height=6, aspect=.75,
    ).despine(left=True)
    st.pyplot(fig1)

    st.write("# EDA with respect to Stay length")
    fig2 = sns.catplot(
    data=amount_df.sort_values('Stay'), x="Stay", y="Admission_Deposit", col='Severity of Illness',
        capsize=.2, palette="YlGnBu_d", errorbar="se",
        kind="point", height=10,width=100, aspect=.75,
    )
    st.pyplot(fig2)

    st.write("""
    👉 From the above graph , we can see that irrespective of the proportion of moderate severity cased, the amount of stay for that category is less as well as the cost of hospitalization is related to severity of illness.

    👉 If someone ,who having minority of illness , want to stay for longer duration , they need to spend more money interms of initial deposite as well as overall money. In contrast , the cost of extreme moderated decease is low compare to other group. This is group.

    👉 On bad thing is there for extreme case, the amount of visitor for them is higher. So It increase the risk of infaction. So that hospital need to regulate.
    """)
    # #ax.bar(df_dept.index.values,height='case_id',color='Department',labels={'Count':'Number of patients'})
    # # # ax.set_title('Case load distribution per department')
    # # # fig.ax = plt.subplot()
    # # # ax.bar()



st.sidebar.write("")
st.sidebar.write("""Developed By 
    **@Kevin Patel**""")
  