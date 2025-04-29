import pickle

import pandas as pd
import streamlit as st

st.title('Work Site Risk Assessment')

with st.form("my_form"):
    st.write("Form")
    choices = []
    # Factor, Exposure, Severity, Likelihood
    questions = [['AADT (Annual Average Daily Traffic)', True, False, False],
                 ['Absence Barricade Boards', False, False, True],
                 ['Absence Road Humps with Road Markings', False, False, True],
                 ['Absence Rumble Strips', True, False, True],
                 ['Absence Traffic Cones', False, False, True],
                 ['Absence Warning Lighting Devices ', False, False, True],
                 ['Absence of Crash Barriers', False, True, False],
                 ['Absence of Edge Markers', False, False, True],
                 ['Absence of Guide Posts or Traffic Guide Guardman', False, False, True],
                 ['Absence of Hazard Markers', False, False, True],
                 ['Absence of Pavement Markings', False, False, True],
                 ['Absence of Pedestrian Fencing', False, True, False],
                 ['Absence of Tapers', False, False, True],
                 ['Activity Area Length', False, False, True],
                 ['Buffer Area Length', False, False, True],
                 ['Climate Conditions', False, True, False],
                 ['Cyclists Traffic', True, False, False],
                 ['Duration of Construction', False, True, False],
                 ['Improper Layout', False, False, True],
                 ['Inadequate Carriageway Width', False, False, True],
                 ['Inadequate Clearance at Bridges and Structures', False, False, True],
                 ['Inadequate Lane Width (3m to 3.6m)', False, False, True],
                 ['Inadequate Lighting', False, False, True],
                 ['Inadequate Passing Opportunities', False, True, False],
                 ['Inadequate Shoulder Width', False, False, True],
                 ['Inadequate Sight Distance', False, True, False],
                 ['Inadequate Visibility Triangle', False, True, False],
                 ['Insufficient Clear Zone', False, False, True],
                 ['Number of Lanes', False, False, True],
                 ['Pavement Defects', False, True, False],
                 ['Pedestrians Traffic', True, False, False],
                 ['Presence of Poles, Objects, or Deep Excavations', False, True, False],
                 ['Presence of Precipices', False, True, False],
                 ['Termination Area Length', False, False, True],
                 ['Transition Area Length', False, False, True],
                 ['Type of Work Zone', False, True, False],
                 ['Unavailability of Warning Signs in Work Zone', False, False, True],
                 ['Work Equipment Use', False, True, False],
                 ['Work Zone Location', False, True, False],
                 ['Work Zone Vehicle Operating Time', False, True, False]]

    no_not_applicable = [1,2,3,4,5,6,7,8,9,10,12,13,15,17,18,19,21,23,24,25,27,28,29,35,36,37,38]
    for i in range(len(questions)):
        q = questions[i]
        if i in no_not_applicable:
            choice = st.selectbox(q[0], ['High', 'Low'], index=None, key=q)
        else:
            choice = st.selectbox(q[0], ['High', 'Low', 'Not Applicable'], index=None, key=q)
        choices.append(choice)

    submitted = st.form_submit_button('Submit')

    if submitted:
        if None in choices:
            st.error('Please select an option for all questions before submitting.')
        else:
            st.success('Form submitted successfully!')
            # Do whatever you want with the data
            X_1 = []
            X_2 = []
            for i in range(len(choices)):
                choice = choices[i]
                if choice == 'Not Applicable':
                    choice = "Not_Applicable"

                name = None
                if questions[i][1]:
                    name = f"Exposure_{choice}"
                    if choice == 'High' or choice == 'Low':
                        X_2.append(1)
                    else:
                        X_2.append(0)
                else:
                    X_2.append(0)

                if questions[i][2]:
                    name = f"Severity_{choice}"
                    if choice == 'High' or choice == 'Low':
                        X_2.append(1)
                    else:
                        X_2.append(0)
                else:
                    X_2.append(0)

                if questions[i][3]:
                    name = f"Likelihood_{choice}"
                    if choice == 'High' or choice == 'Low':
                        X_2.append(1)
                    else:
                        X_2.append(0)
                else:
                    X_2.append(0)

                for row_name in ['Exposure_High', 'Exposure_Low', 'Exposure_Not_Applicable', 'Severity_Major',
                                 'Severity_Minor', 'Severity_Not_Applicable', 'Likelihood_High', 'Likelihood_Low',
                                 'Likelihood_Not_Applicable']:
                    if name == row_name:
                        X_1.append(1)
                    else:
                        X_1.append(0)
            print(X_1, len(X_1))
            print(X_2, len(X_2))

            input_X_1 = [X_1]
            input_X_2 = [X_2]

            # ['High' 'Low' 'Medium'] => 0, 1, 2

            results = {}
            with open('models/decision_tree_1.pkl', 'rb') as f:
                decision_tree_1 = pickle.load(f)
                output = decision_tree_1.predict(input_X_1)
                print("decision_tree_1", output)
                results['Decision Tree 1'] = output[0]

            with open('models/decision_tree_2.pkl', 'rb') as f:
                decision_tree_2 = pickle.load(f)
                output = decision_tree_2.predict(input_X_2)
                print("decision_tree_2", output)
                results['Decision Tree 2'] = output[0]

            with open('models/random_forest_classifier_1.pkl', 'rb') as f:
                random_forest_classifier_1 = pickle.load(f)
                output = random_forest_classifier_1.predict(input_X_1)
                print("random_forest_classifier_1", output)
                results['Random Forest Classifier 1'] = output[0]

            with open('models/random_forest_classifier_2.pkl', 'rb') as f:
                random_forest_classifier_2 = pickle.load(f)
                output = random_forest_classifier_2.predict(input_X_2)
                print("random_forest_classifier_2", output)
                results['Random Forest Classifier 2'] = output[0]

            with open('models/xgb_classifier_1.pkl', 'rb') as f:
                xgb_classifier_1 = pickle.load(f)
                output = xgb_classifier_1.predict(input_X_1)
                print("xgb_classifier_1", output)
                results['XGBoost Classifier 1'] = output[0]

            with open('models/xgb_classifier_2.pkl', 'rb') as f:
                xgb_classifier_2 = pickle.load(f)
                output = xgb_classifier_2.predict(input_X_2)
                print("xgb_classifier_2", output)
                results['XGBoost Classifier 2'] = output[0]

            table_view = True

            if table_view:
                # ==================== Stylish Table View =================================
                results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Prediction_Label'])
                results_df['Prediction'] = None
                for index, row in results_df.iterrows():
                    if row['Prediction_Label'] == 0:
                        results_df.loc[index, 'Prediction'] = 'High Risk'
                    elif row['Prediction_Label'] == 1:
                        results_df.loc[index, 'Prediction'] = 'Low Risk'
                    elif row['Prediction_Label'] == 2:
                        results_df.loc[index, 'Prediction'] = 'Medium Risk'
                    else:
                        raise Exception("Error")

                # Show as a nice table
                st.subheader("📝 Model Predictions")
                st.dataframe(results_df[['Model', 'Prediction']], use_container_width=True)
            else:
                # ================== Fancy Card View =====================================
                st.subheader("🚦 Model Predictions")

                for model_name, prediction in results.items():
                    # Choose a color based on prediction value
                    if prediction == 0:
                        color = 'red'
                        prediction_text = "High Risk"
                    elif prediction == 1:
                        color = 'green'
                        prediction_text = "Low Risk"
                    elif prediction == 2:
                        color = 'yellow'
                        prediction_text = "Medium Risk"
                    else:
                        raise Exception("Error")

                    # Make a colored container
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="background-color: {color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                                <h4 style="color: black;">{model_name}</h4>
                                <h2 style="color: black;">Prediction: {prediction_text}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
