
import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from joblib import load
from PIL import Image
from tensorflow.keras import Input
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Load the model
print("Loading the model file...")
loaded_model = tf.keras.models.load_model('role_skills_descriptions_trained_model.keras') 
print("Model loaded.")

# Load the LabelEncoder
print("Loading LabelEncoder...")
label_encoder = load('role_skills_descriptions_trained_model.joblib')
print("LabelEncoder loaded.")

# Load the pre-trained SBERT model
print("Loading SBERT model for inference...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("SBERT model loaded.")

def predict_top_5_job_roles(skills_text):
    """
    Predicts the top 5 most probable job roles based on the given skills.

    Args:
    skills_text: A string containing the skills of the individual.

    Returns:
    A list of tuples, where each tuple contains the predicted job role and its probability.
    """
    print('Creating skill embeddings...')
    # Preprocess the skills text
    skill_embedding = sbert_model.encode(skills_text)

    # Reshape the embedding for TensorFlow model input
    input_data = tf.reshape(skill_embedding, (1, -1)) 
    input_data = tf.cast(input_data, dtype=tf.float32) 

    print('Generating predictions using model...')
    # Use the Keras model to generate predictions, not the SBERT model
    predictions = loaded_model.predict(input_data)

    print('Getting the top 5 indices...')
    # Get the top 5 indices of the highest probabilities
    top_5_indices = np.argsort(predictions[0])[::-1][:5] 

    print('Decoding the labels into role names...')
    # Decode the labels back to role names
    top_5_roles = label_encoder.inverse_transform(top_5_indices)

    print('Getting the probabilities for top 5 roles...')
    # Get the probabilities for the top 5 roles
    top_5_probabilities = predictions[0][top_5_indices]

    print('Adding top 5 roles to a list...')
    # Create a list of tuples containing the role and its probability
    predicted_roles = [(role, prob) for role, prob in zip(top_5_roles, top_5_probabilities)]

    return predicted_roles

# Configure Streamlit app settings
st.set_page_config(
    page_title="SkillX",
    page_icon="✖️",
    layout="centered",
)

# Load and display the app logo
LOGO = "SkillX.jpg"
image = Image.open(LOGO)

container1 = st.container(border=True)
with container1:
    st.image(image, caption="", clamp=False, channels="RGB", use_container_width=True)

st.logo(LOGO, link=None, icon_image=None)

# Set up the generative AI library with your API key
#GOOGLE_API_KEY = 'AIzaSyBR7ZJaCfTKChDWnZJLJTqS20o9OCX_YuQ'
GOOGLE_API_KEY = 'AIzaSyAfxmrmwhu62afqajL84gI5hta6LDUs9yc'
genai.configure(api_key=GOOGLE_API_KEY)

# Disclaimer
expander = st.expander("Disclaimer")
expander.write('''
    This is a prototype under development and may contain bugs or errors. 
    It is intended for testing and educational purposes only. 
    Please use this prototype with caution and at your own risk.
''')

def download_message_history():
    message_history = "\n".join([msg["content"] for msg in st.session_state.messages])
    st.download_button(
        label="Download Recommendations",
        data=message_history,
        file_name="your_career_plan.txt",
        mime="text/plain"
    )

# Initialize chat session state
container = st.container(border=True)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Form to collect user data
with st.form("user_profile"):
    st.markdown("""
        <div> 
            <h3 style="color:#48acd2; text-align:center;">✨ Discover your next career move with SkillX! ✨</h3>
            <p style="font-size:16px; line-height:1.6;">
                🚀 Our <span style="color:#0078D4;">AI technology</span> analyzes <strong>1.5 million jobs </strong> 
                and <strong>skills</strong> to pinpoint roles that are a match for <em><strong><span style="color:#4CAF50;">you</span></strong></em>.
            </p>
            <p style="font-size:16px; line-height:1.6;">
                💡 We'll show you how your <span style="color:#FF9800;">skills align</span> with each role and provide 
                detailed job descriptions.
            </p>
            <p style="font-size:16px; line-height:1.6;">
                👉 To begin, I'll need a little information from you. Let’s get started!
            </p>
        </div>
    """, unsafe_allow_html=True)

    current_role = st.text_input("What is your current role?", placeholder=None)
    top5_skills = st.text_input("Please provide your top 5 skills (separate with commas):")
    submitted = st.form_submit_button("Submit")

# Define a @tf.function outside to avoid retracing
#@tf.function(reduce_retracing=True)
#def predict_roles(keras_model, input_embedding):
    #return keras_model(input_embedding, training=False)

# Process form submission
if submitted:
    role_predictions = predict_top_5_job_roles(top5_skills)

    print('Creating prompt for gemini...')
    # Prepare the prompt
    career_prompt = f"""You are a career advisor specializing in helping professionals transition to new roles. You will receive information about a user's current job title and their top 5 skills, and a list of target job titles they wish to transition into. Your task is to provide insightful and concise guidance to help them understand the alignment between their skills and their desired roles. For each target role, provide a skill match analysis summarizing the *overall* match of the user's skillset to the demands of the role, considering all 5 skills together, and then a detailed breakdown of each skill.

        Here is the user's information:

        Current Role: {current_role}
        Top 5 Skills: {(top5_skills)}
        Target Jobs: {(role_predictions)}

        Do not output the response in JSON. Instead, output the response in the structure listed below in the "Roles to Consider" object for all 5 provided target jobs. For the "Skill match analysis explained" you must present the skill match analysis in a table for each of the 5 skills. Don't forget to provide recommendations for developing the skill. If any of the "Target Jobs" are erroneous or not applicable to the top 5 skills provided, please select a job title to replace it from your model without annotating the job was replaced next to the job role. After the 5 jobs, at the very end of the result provide a list of any jobs you replaced.

          "Roles to consider":
              "Matched Job Title #: Inser Job Title here",
              "Job description": "Concise and informative job description (around 2-3 sentences). Focus on key responsibilities and required skills.",
              "Average Annual Sallary: "If available, this is as an average annual salary for the role. If salary information is unavailable, please indicate it is unavailable.",
              "Skill match analysis summary": "Overall analytical assessment of how well the user's *entire skillset* (all 5 skills combined) aligns with the requirements of this role. Consider the relative importance of each skill to the role. Examples: 'Strong overall match...', 'Partial match...', 'Weak overall match...'",
              "Skill match analysis explained":
                  "Skill": "Skill Name",
                  "Match": "Weak/Medium/Strong",
                  "Justification": "Brief explanation."
              "How your skills apply to this role": "Explain how the user's provided skills can be practically applied to this specific job. Provide concrete examples and focus on transferable skills. Aim for 3-4 sentences."
              "Skill Development": "Provide recommendations to the user to help them develop this skill."
        """
    print('Retrieving response from Gemini..')
    with st.spinner("Generating Google Gemini response..."):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-002")
            response = model.generate_content(career_prompt)

            response_text = ""
            for chunk in response:
                try:
                    text_content = chunk.candidates[0].content.parts[0].text
                    response_text += text_content
                except (KeyError, IndexError) as e:
                    st.error(f"Error extracting chunk text: {e}")

            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display chat messages
container2 = st.container(border=True)
if st.session_state.messages:
    for message in st.session_state.messages:
        with container2.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.write("No messages in history yet.")

# Add download button if there are messages
if st.session_state.messages:
    download_message_history()