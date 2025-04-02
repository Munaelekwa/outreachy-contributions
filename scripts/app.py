import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import streamlit as st

# Load the trained model using joblib
model = joblib.load("models/bbbp_model.pkl")

# Load selected feature names used during training
selected_features = joblib.load("selected_features.pkl")

# Function to compute RDKit descriptors for a given SMILES string
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Handle invalid SMILES strings gracefully

    descriptor_names = [desc[0] for desc in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # Compute descriptor values
    descriptors = np.array(calc.CalcDescriptors(mol)).reshape(1, -1)

    # Convert to DataFrame
    descriptors_df = pd.DataFrame(descriptors, columns=descriptor_names)

    # Ensure selected features are available
    missing_features = set(selected_features) - set(descriptors_df.columns)
    if missing_features:
        st.error(f"Error: Missing features {missing_features}")
        return None

    return descriptors_df[selected_features]  # Return only selected features

# Streamlit UI
st.set_page_config(page_title="BBB Permeability Predictor", layout="centered")

st.title("🧪 BBB Permeability Predictor")
st.write("Predict whether a compound can cross the blood-brain barrier using machine learning.")

# Input box for SMILES
smiles_input = st.text_input("Enter a **SMILES** string:", "")

if st.button("Predict"):
    if smiles_input:
        with st.spinner("Analyzing molecule..."):
            descriptors = compute_descriptors(smiles_input)
            if descriptors is not None:
                prediction = model.predict(descriptors)[0]
                probability = model.predict_proba(descriptors)[0][1]

                st.success(f"✅ Prediction: **{'This compound is BBB Permeable' if prediction == 1 else 'This compound is Non-Permeable'}**")
                st.write(f"**Confidence Score:** {probability:.2f}")
            else:
                st.error("❌ Invalid SMILES notation. Please check your input.")
    else:
        st.warning("⚠️ Please enter a SMILES string before predicting.")

# Footer
st.markdown("---")
st.write("🔬 Developed to support drug discovery research.")
st.write("👨‍💻 Built by **Munachi Elekwa**")

