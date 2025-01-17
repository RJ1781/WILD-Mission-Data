import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Charger les fichiers CSV
df = pd.read_csv('ML_EDA.csv')
df2 = pd.read_csv('ML_EDA.csv')
precautions_df = pd.read_csv("Disease precaution.csv")

# Nettoyage des données (similaire à votre code)
df = df.drop(columns=['Disease'])
df.columns = df.columns.str.strip()

# Diviser les données en X (symptômes) et y (maladies)
X = df.drop(columns=['Disease_Encode'])
y = df['Disease_Encode']

# Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer l'instance du classificateur KNN avec un nombre de voisins (par exemple, k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Liste des symptômes
symptoms_list = ['Unnamed: 74', 'abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure', 'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads', 'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool', 'blurred_and_distorted_vision', 'breathlessness', 'brittle_nails', 'bruising', 'burning_micturition', 'chest_pain', 'chills', 'cold_hands_and_feets', 'coma', 'congestion', 'constipation', 'continuous_feel_of_urine', 'continuous_sneezing', 'cough', 'cramps', 'dark_urine', 'dehydration', 'depression', 'diarrhoea', 'dischromic _patches', 'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips', 'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 'foul_smell_of urine', 'headache', 'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching', 'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance', 'loss_of_smell', 'malaise', 'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes', 'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 'palpitations', 'passage_of_gases', 'patches_in_throat', 'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness', 'runny_nose', 'rusty_sputum', 'scurring', 'shivering', 'silver_like_dusting', 'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails', 'spinning_movements', 'spotting_ urination', 'stiff_neck', 'stomach_bleeding', 'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremeties', 'swollen_legs', 'throat_irritation', 'toxic_look_(typhos)', 'ulcers_on_tongue', 'unsteadiness', 'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs', 'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze', 'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin']

# Fonction pour transformer les symptômes de l'utilisateur en une ligne d'entrée pour le modèle
def transform_symptoms_to_features(user_symptoms, symptoms_list):
    features = [0] * len(symptoms_list)  # Crée un vecteur de zéros
    for symptom in user_symptoms:
        index = symptoms_list.index(symptom)
        features[index] = 1  # On indique que ce symptôme est présent
    return np.array(features).reshape(1, -1)

# Fonction pour afficher les précautions associées à une maladie
def afficher_precautions(maladie_predite, precautions_df):
    if maladie_predite in precautions_df['Disease'].values:
        prec_row = precautions_df[precautions_df['Disease'] == maladie_predite]
        precautions = prec_row.iloc[0, 1:]  # Extraire toutes les précautions
        precautions = precautions.dropna()  # Supprimer les NaN
        return precautions.tolist()
    else:
        return []

# Interface utilisateur Streamlit
st.title("Pré-Diagnostic Médical Virtuel")

# Permet à l'utilisateur de sélectionner les symptômes
selected_symptoms = st.multiselect("Veuillez sélectionnez 5 symptômes parmi la liste suivante :", symptoms_list, max_selections=5)

# Vérifier si l'utilisateur a sélectionné exactement 5 symptômes
if len(selected_symptoms) < 5:
    st.warning("Veuillez sélectionner exactement 5 symptômes pour activer le diagnostic.")
elif len(selected_symptoms) == 5:
    if st.button("Obtenir le Diagnostic"):
        # Transformer les symptômes en vecteur de caractéristiques
        user_features = transform_symptoms_to_features(selected_symptoms, symptoms_list)

        # Prédire la maladie
        predicted_disease_code = knn.predict(user_features)[0]
        maladie_predite = df2[df2['Disease_Encode'] == predicted_disease_code]['Disease'].iloc[0]
        
        # Afficher la maladie prédite
        st.subheader(f"Maladie prédite par le Pré-Diagnostic Médical Virtuel : {maladie_predite}")

        # Afficher les précautions associées
        precautions = afficher_precautions(maladie_predite, precautions_df)
        if precautions:
            st.write("Mesures préventives conseillées :")
            for i, precaution in enumerate(precautions, 1):
                st.write(f"{i}. {precaution}")
        else:
            st.write("Aucune précaution trouvée pour cette maladie.")
