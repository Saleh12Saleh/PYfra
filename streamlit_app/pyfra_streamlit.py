import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
from joblib import load
from sklearn.pipeline import Pipeline

def run():
    st.set_page_config(
        page_title="PYFRA",
        page_icon=":car:",
    )

    pages = ["Project Presentation","Data Introduction and Cleaning","Data Visualization","Modeling","Conclusion"]
    with st.sidebar:
        st.write('# Overview')
        page = st.radio(label='',options=pages)
        is_presentation = st.checkbox(label='Presentation Mode', value=True)

    if page==pages[0]:
        st.header("PYFRA")
        st.write("# Car Accident Analysis in France")
        st.image(image='images/stable_diffusion.jpeg', 
                width=200,
                caption='Generated with Stable Diffusion')
        
        if not is_presentation:
            st.markdown(
                """
                *Objective of this presentation is to explain the summary and objective of the project
                *Full analysis and steps exist in the different notebooks
                
                ### Github
                [Project Page](https://github.com/DataScientest-Studio/pyfra)
                """)
        st.markdown("""
            ### Project Members
            * Kay Langhammer
            * Robert Leisring
            * Saleh Saleh
            * Michal Turák

            ### Project Mentors
            * Robin Trinh
            * Laurene Bouskila
        """
        )

    if page==pages[1]:
        st.header('Data Introduction and Cleaning')
        

        st.subheader('Data Source')

        if not is_presentation:
            st.write('''
            The data used for this analysis is found in the official French goverment records and is split
            each year into 4 categories that include csv files of: Users, Characteristics, Vehicules, and Location.

            [Goverment Database](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/)
            ''')

        st.components.v1.html(height=300, width=2500, html="""
        <style type="text/css">
        .tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
        .tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
        font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
        .tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
        font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
        .tg .tg-0lax{text-align:left;vertical-align:top}
        </style>
        <table class="tg">
        <thead>
        <tr>
            <th class="tg-0lax" colspan="3">Characteristics</th>
            <th class="tg-0lax" colspan="2">Places</th>
            <th class="tg-0lax" colspan="3">Vehicles</th>
            <th class="tg-0lax" colspan="3">Users</th>
        </tr>
                <tr>
            <td class="tg-0lax">Accident ID</td>
            <td class="tg-0lax">Weather</td>
            <td class="tg-0lax">Collision <br> Type</td>
            <td class="tg-0lax"># Lanes</td>
            <td class="tg-0lax">Road Condition</td>
            <td class="tg-0lax">Vehicle ID</td>
            <td class="tg-0lax"># Passengers</td>
            <td class="tg-0lax">Obstacle <br> type</td>
            <td class="tg-0lax">Sex</td>
            <td class="tg-0lax">Year of <br> Birth</td>
            <td class="tg-0lax">Severity</td>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td class="tg-0lax" rowspan="5">01</td>
            <td class="tg-0lax" rowspan="5">Light Rain</td>
            <td class="tg-0lax" rowspan="5">Frontal</td>
            <td class="tg-0lax" rowspan="5">2</td>
            <td class="tg-0lax" rowspan="5">Wet</td>
            <td class="tg-0lax" rowspan="3">01</td>
            <td class="tg-0lax" rowspan="3">3</td>
            <td class="tg-0lax" rowspan="3">Vehicle</td>
            <td class="tg-0lax">Male</td>
            <td class="tg-0lax">1985</td>
            <td class="tg-0lax">Hospitalized</td>
        </tr>
        <tr>
            <td class="tg-0lax">Female</td>
            <td class="tg-0lax">1988</td>
            <td class="tg-0lax">Unharmed</td>
        </tr>
        <tr>
            <td class="tg-0lax">Female</td>
            <td class="tg-0lax">1960</td>
            <td class="tg-0lax">Unharmed</td>
        </tr>
        <tr>
            <td class="tg-0lax" rowspan="2">02</td>
            <td class="tg-0lax" rowspan="2">2</td>
            <td class="tg-0lax" rowspan="2">Vehicle</td>
            <td class="tg-0lax">Male</td>
            <td class="tg-0lax">1972</td>
            <td class="tg-0lax">Unharmed</td>
        </tr>
        <tr>
            <td class="tg-0lax">Female</td>
            <td class="tg-0lax">1979</td>
            <td class="tg-0lax">Unharmed</td>
        </tr>

        </tbody>
        </table>
        """)

        st.write('''

        ### Data import and cleaning
        * Introduction to the dataset
        * Features renaming and fixing inconsistency 
        * Handling missing values
        * One-Hot Encoding of Categorical Features
        

        * The Data present was of large volume
        * There was incoherence in the variables across the csv files and years
        * A high percentage of missing values


        ''')
        st.header('Data improvement for modeling')

        #image = Image.open("./figures/Nan.png")
        st.image("figures/Nan.png")

    if page==pages[2]:
        st.write('# Data Visualization')
        subpages = ['Hypothesis 1', 'Hypothesis 2', 'Hypothesis 3', 'Hypothesis 4' , 'Hypothesis 5' , 'Hypothesis 6']
        subpage = st.sidebar.radio(label='', options=subpages)

        if subpage==subpages[0]:
            st.header("Hypothesis 1  : Generally Accidents rarely result in serious injury or death")
            image3 = Image.open("./figures/Nb Accidents by Severity.png")
            st.image(image3)
        
        if subpage==subpages[1]:
            st.header("Hypothesis 2  : Number of Accidents generally should increase by year due to higher volume of traffic")
            image6 = Image.open("./figures/Accidents per Year.png")
            st.image(image6)
        
        if subpage==subpages[2]:
            st.header("Hypothesis 3  : Number of Accidents are higher for bad weather conditions")
            image5 = Image.open("./figures/Accidents by Weather conditions.png")
            st.image(image5)


        if subpage==subpages[3]:
            st.header("Hypothesis 4 : Accidents occur more often during a Specific time of the day and on weekends")
            image = Image.open("./figures/Accidents by Daytime.png")
            st.image(image)

        if subpage==subpages[4]:
            st.header("Hypothesis 5  : Accidents are split evenly among all age groups")
            image2 = Image.open("./figures/Accidents per Age.png")
            st.image(image2)
       
    

        if subpage==subpages[5]:
            st.header("Hypothesis 6  : Severity of Accidents are split uniformly across all ages and sex groups")
            image4 = Image.open("./figures/Violin Chart.png")
            st.image(image4)

    

    



    if page==pages[3]:
        st.write('# Modeling')
        subpages = ['Model Comparison', 'Simulation', 'Impact of Input Size', 'Impact of Data Imbalance']
        subpage = st.sidebar.radio(label='', options=subpages)

        if subpage==subpages[0]:
            st.subheader(subpages[0])
            st.markdown("""
            * 4 Basic Models
            * 2 Advanced Models
            """)
            result_metrics = pd.read_pickle('./data/nb_3_results.p')

            st.subheader('$f_1$ Score by Model')
            res_chart = alt.Chart(result_metrics).mark_bar().encode(
                x='f1',
                y="model"
            ).properties(height=300, width=500)
            
            # Plot the f1 scores
            st.altair_chart(res_chart)

            # Plot confusion matrix
            st.subheader('Confusion Matrices for each Model')
            tab_svc, tab_log_reg, tab_dt, tab_rf, tab_ada, tab_stacking = \
                st.tabs(['SVC', 'Logistic Regression', 'Decision Tree', 
                         'Random Forest', 'AdaBoost','Stacking'])
            with tab_svc:
                st.image('figures/svc_conf.png')
            with tab_log_reg:
                st.image('figures/log_reg_conf.png')
            with tab_dt:
                st.image('figures/dt_conf.png')
            with tab_rf:
                st.image('figures/rf_conf.png')                
            with tab_ada:
                st.image('figures/ada_conf.png')                                                
            with tab_stacking:
                st.image('figures/stacking_conf.png')

            # Display the metrics table
            #st.dataframe(data=result_metrics)



        if subpage==subpages[1]:
            st.subheader(subpages[1])
            log_reg_preprocessing_pipe = load('models/log_reg_preprocessing_pipeline.joblib')
            log_reg_clf = load('models/log_reg_nb5.joblib')

            # Select Department
            date = st.date_input("Select Date", datetime.date(2022,2,14))
            time = st.time_input('Select time', datetime.time(18, 0))
            weather = st.selectbox('Select weather', options=('Normal', 'Light Rain', 'Heavy Rain', 'Snow', 'Fog'))
            year_of_birth = st.number_input('Select birthyear', min_value=1930, max_value=2022)
            car_construct_year = st.number_input('Select year of car construction', min_value=1950, max_value=2022)
            collision_type = st.selectbox('Select collision', options=['Two vehicles -- frontal', 
                                                                       'Two vehicles -- rear-end',
                                                                       'Two vehicles -- from the side',
                                                                       'Three or more vehicles in a chain'])
            n_sample = 1000
            st.write('Result (based on', n_sample, ' data points)')
            example_df = pd.read_pickle('data/streamlit_example.p').sample(n_sample)
            example_df['YoB'] = year_of_birth
            example_df['year'] = date.year
            example_df['month'] = date.month
            example_df['day'] = date.day
            example_df['hhmm'] = int(str(time.hour) + str(time.minute))
            example_df['day_of_week'] = date.weekday()
            
            if date.weekday() < 6:
                example_df['is_weekend'] = 0
            else:
                example_df['is_weekend'] = 1
            
            # Atmospheric Conditions
            atmo_columns = ['atmospheric_conditions_'+str(i)+'.0' for i in range(1,10)]
            example_df[atmo_columns] = 0
            if weather =='Normal':
                example_df['atmospheric_conditions_1.0'] = 1
            if weather == 'Light Rain':
                example_df['atmospheric_conditions_2.0'] = 1
            if weather == 'Heavy Rain':
                example_df['atmospheric_conditions_3.0'] = 1
            if weather == 'Snow':
                example_df['atmospheric_conditions_4.0'] = 1
            if weather == 'Fog':
                example_df['atmospheric_conditions_5.0'] = 1

            # Collision Type
            collision_columns = ['collision_category_'+str(i)+'.0' for i in range(1,8)]
            example_df[collision_columns] = 0
            if collision_type =='Two vehicles -- frontal':
                example_df['collision_category_1.0'] = 1
            if collision_type == 'Two vehicles -- rear-end':
                example_df['collision_category_2.0'] = 1
            if collision_type == 'Two vehicles -- from the side':
                example_df['collision_category_3.0'] = 1
            if collision_type == 'Three or more vehicles in a chain':
                example_df['collision_category_4.0'] = 1         

            #st.dataframe(data=example_df)
            example_df = log_reg_preprocessing_pipe.transform(example_df)
            #st.write(log_reg_clf.feature_names_in_)
            log_reg_pred, log_reg_pred_counts = np.unique(log_reg_clf.predict(example_df), return_counts=True)
            #st.write(len(svc_pred_counts))
            predictions_df = pd.DataFrame(index=log_reg_pred, data={'P log. Reg.': log_reg_pred_counts*100/n_sample}) \
                                        .sort_values(by='P log. Reg.', ascending=False)
            predictions_df.loc[:,'P log. Reg.'] = predictions_df.loc[:,'P log. Reg.'].apply(func=(lambda p: str(p)+' %'))
            predictions_df.rename(index={1: 'Unharmed', 2: 'Killed', 3: 'Hospitalized', 4: 'Lightly Injured'}, inplace=True)
            st.dataframe( predictions_df )
            

        if subpage==subpages[2]:
            st.subheader(subpages[2])
            tab1, tab2 = st.tabs(['Rows', 'Columns'])
            with tab1:
                st.image('figures/n_rows_f1.png')
            with tab2:
                st.image('figures/k_features_f1.png')
                

        if subpage==subpages[3]:
            st.write('## Remodeling with Imbalanced Datasets')
            

            st.write('#### Model Comparison')
            result_metrics = pd.read_pickle('./data/nb_4_results.p')

            st.write('## $f_1$ score by model')
            res_chart = alt.Chart(result_metrics).mark_bar().encode(
                x='f1',
                y="model"
            ).properties(height=300, width=500)
            
            # Plot the f1 scores
            st.altair_chart(res_chart)

            # Plot confusion matrix
            st.subheader('Confusion Matrix for imbalanced dataset')
            st.image('figures/Imb_LR.png')

    if page==pages[4]:
        st.write('# Conclusion')
        '''
        * Problems of the data (Size,Incoherence,missing data(Blood Alcohol levels))
        * Data Cleaning and Visualization to understand the project
        * Several modeling techniques to get best results
        * Deep learning model



        '''


if __name__ == "__main__":
    run()


