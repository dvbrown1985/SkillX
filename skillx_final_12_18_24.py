#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# Load the model
print("Loading the model file...")
loaded_model = tf.keras.models.load_model('role_skills_descriptions_industries_trained_model.keras') 
print("Model loaded.")

# Load the LabelEncoder
print("Loading LabelEncoder...")
label_encoder = load('role_skills_descriptions_industries_trained_model.joblib')
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
    st.info('Embedding your inputs...')
    skill_embedding = sbert_model.encode(skills_text)
    input_data = tf.convert_to_tensor(skill_embedding, dtype=tf.float32)
    input_data = tf.expand_dims(input_data, axis=0)

    st.info('Generating predictions from the model...')
    predictions = loaded_model.predict(input_data)
    top_5_indices = np.argsort(predictions[0])[::-1][:5]
    top_5_roles = label_encoder.inverse_transform(top_5_indices)
    top_5_probabilities = predictions[0][top_5_indices]

    predicted_roles = [(role, prob) for role, prob in zip(top_5_roles, top_5_probabilities)]
    st.success('Success!', icon="🤖")
    return predicted_roles

def download_message_history():
    message_history = "\n".join([msg["content"] for msg in st.session_state.messages])
    st.download_button(
        label="Download Recommendations",
        data=message_history,
        file_name="your_career_plan.txt",
        mime="text/plain"
    )

# Set up the generative AI library with your API key
GOOGLE_API_KEY = 'AIzaSyAfxmrmwhu62afqajL84gI5hta6LDUs9yc'
genai.configure(api_key=GOOGLE_API_KEY)

LOGO = "SkillX.jpg"
image = Image.open('SkillX.jpg')

st.logo(LOGO)

# Configure Streamlit app settings
st.set_page_config(
    page_title="SkillX",
    page_icon="✖️",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

container1 = st.container(border=True)
with container1:
    st.image(image, caption=None)

# Disclaimer
expander = st.expander("Disclaimer")
expander.write('''
    This is a prototype under development and may contain bugs or errors. 
    It is intended for testing and educational purposes only. 
    Please use this prototype with caution and at your own risk.
''')

container_x = st.container(border=True)
with container_x:
    st.markdown("""
    <div> 
        <h3 style="color:#48acd2; text-align:center;">Discover your next career move with Skill(X)</h3>
        <p style="font-size:17px; line-height:1.6;">
            Our AI technology analyzes <strong>1.5 million jobs </strong> 
            and <strong>skills</strong> to pinpoint roles that are a match for <strong>you</strong>.
        </p>
        <p style="font-size:17px; line-height:1.6;">
            We'll show you how your skills align with each role and provide 
            detailed job descriptions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Define the process_submission function
def process_submission(submitted, career_goal, industry, top_skills):
    """
    Process the form data when submitted.
    """
    if submitted:
        # You can add any additional processing logic here if necessary
        print(f"Processing form data: {career_goal}, {industry}, {top_skills}")
        return True
    return False

with st.form("user_profile"):
    industries = [
        'Aerospace & Defense', 'Advertising and Marketing', 'Aluminum and Copper Manufacturing', 'Apparel/Fashion',
        'Automotive', 'Banking & Financial Services', 'Beverages', 'Business Services - Data/Analytics', 'Business Services - Pest Control',
        'Business Services - Testing/Compliance', 'Cement and Building Materials', 'Chemicals', 'Cloud Software and Services',
        'Commercial Banks', 'Computer Software', 'Construction', 'Construction & Building Materials', 'Construction and Engineering',
        'Consumer Goods', 'Consumer Products - Cleaning Products', 'Consumer Products - Cosmetics', 'Consumer Products - Food',
        'Consumer Products - Food & Beverages', 'Consumer Products - Food Production', 'Diversified',
        'Diversified Outsourcing Services', 'E-commerce', 'E-commerce & Technology', 'E-commerce/Technology', 'Electronics',
        'Energy', 'Energy - Coal', 'Energy - Diversified', 'Energy - Oil & Gas Exploration & Production', 'Energy - Oil & Gas Services',
        'Energy - Utilities', 'Energy - Oil & Gas', 'Entertainment', 'Entertainment - Music Streaming', 'Entertainment - Streaming Services',
        'Entertainment - Satellite Radio', 'Food and Drug Stores', 'Food & Beverage', 'Food Manufacturing',
        'General Merchandisers', 'Gambling', 'Health Care: Insurance and Managed Care', 'Health Care: Medical Facilities',
        'Health Care: Pharmacy and Other Services', 'Healthcare', 'Hospitality & Entertainment', 'Hospitality/Hotels',
        'Hotels, Casinos, Resorts', 'Industrial Conglomerate', 'Industrial Machinery', 'Industrial Manufacturing', 'Insurance',
        'Insurance: Life, Health (Stock)', 'Insurance: Property and Casualty', 'Insurance: Property and Casualty (Mutual)',
        'Information Technology', 'Information Technology Services', 'Information Technology and Services', 'Investment Banking',
        'Investment Management', 'Investment Services', 'Jewelry and Watches', 'Logistics', 'Logistics & Delivery Services',
        'Manufacturing', 'Manufacturing & Transportation', 'Manufacturing - Diversified', 'Manufacturing/Steel', 'Media and Entertainment',
        'Metals', 'Mining', 'Oil and Gas', 'Oil and Gas Equipment, Services', 'Oil & Gas', 'Pipelines', 'Personal Care & Cosmetics',
        'Pharmaceuticals', 'Ports and Infrastructure', 'Publishing, Printing', 'Renewable Energy', 'Restaurants', 'Retail',
        'Retail - Discount/Department Stores', 'Retail - Electronics', 'Retail - Farm and Ranch Supplies', 'Retail - Food & Drug',
        'Retail - General Merchandise', 'Retail - Home Improvement', 'Retail - Books & News', 'Specialty Retailers: Apparel',
        'Specialty Retailers: Other', 'Steel', 'Securities', 'Technology & Electronics', 'Technology & Social Media', 'Technology & Telecommunications',
        'Telecommunications', 'Transportation', 'Transportation - Rail', 'Transportation - Airlines', 'Transportation - Logistics',
        'Transportation Equipment', 'Transportation/Infrastructure', 'Transportation/Logistics', 'Trucking, Truck Leasing',
        'Utilities', 'Waste Management', 'Wholesalers: Electronics and Office Equipment', 'Wholesalers: Food and Grocery',
        'Wholesalers: Diversified', 'Wholesalers: Health Care', 'Travel and Leisure - Hotels',
        'Travel and Leisure - Cruises', 'Technology & Search Engines', 'Tire Manufacturing', 'Tobacco', 'Technology & Entertainment'
    ]
    industries.sort()
    career_goal = ['I want to be promoted', 'I want to transition into a new role']

    st.info("Let's create your profile! Please provide your career goal, industry, current role, and top skills.")

    collect_career_goal = st.selectbox("What kind of professional growth are you looking for?",
                                       options=career_goal,
                                       index=None,
                                       placeholder="Select...")
    
    collect_industry = st.selectbox("What is your current industry?",
                                    options=industries,
                                    index=None,
                                    placeholder="Select...")
    
    current_role = st.text_input("What is your current role?", placeholder=None)
    top5_skills = st.text_input("Please provide your top 5 skills:")

    # Place the submit button inside the form
    submitted = st.form_submit_button("Submit")
    
    # Process the form only when submitted and validate the inputs
    if submitted and not st.session_state.form_submitted:
        # Check if all required fields are filled
        if not collect_career_goal or not collect_industry or not current_role or not top5_skills:
            st.error("Please fill in all the fields before submitting.")
        else:
            st.session_state.form_submitted = True  # Mark form as submitted
            st.success('Thanks for submitting!')  # Show a thank-you message after submission
            
            if collect_career_goal == "I want to be promoted":
                prompt = f"""You are a career advisor specializing in helping professionals qualify and develop skills needed to qualify for a promotion. 

                You will receive information about a user's current job title, their top 5 skills, their current industry. 

                Their current job title is: {current_role}.
                Their current industry is: {collect_industry}
                Their top 5 Skills are: {(top5_skills)}

                You need to identify 3 roles they can be promoted to. One role must be a manager role, One role must be a Director role, and 
                One role must be an executive leader role.
                You also need to explain how their top 5 skills apply to the roles you've identified. 
                You need to identify any missing critical skills for the identified roles not included in their top 5 skills. 

                Do not output the response in JSON and make sure your response uses "You, Your, You're" when referring to the job seeker. ** DO NOT USE THE TERMS "Their or They're." Output the response in the structure listed below in the "Roles to Consider" object for all 5 identified jobs. For the "Skill match analysis explained" you must present the skill match analysis in a table for each of the 5 skills. Don't forget to provide recommendations for developing the skill.

                "Your recommendations":
                      "** Recommendation #: Insert Job Title here ** MAKE THE JOB TITLE BOLD AND ENSURE IT STANDS OUT FROM THE REST OF THE TEXT **",
                      "Job description": "Concise and informative job description (around 2-3 sentences). Focus on key responsibilities and required skills.",
                      "Average Annual Sallary: "If available, this is as an average annual salary in US dollars for the role. If salary information is unavailable, please indicate it is unavailable.",
                      "Skill match analysis summary": "Overall analytical assessment of how well the user's *entire skillset* (all 5 skills combined) aligns with the requirements of this role. Consider the relative importance of each skill to the role. Examples: 'Strong overall match...', 'Partial match...', 'Weak overall match...'",
                      "Skill match analysis explained":
                          "Skill": "Skill Name",
                          "Match": "Weak/Medium/Strong",
                          "Justification": "Brief explanation."
                      "How their skills apply to this role": "Explain how the user's provided skills can be practically applied to this specific job. Provide concrete examples and focus on transferable skills. Aim for 3-4 sentences."
                      "Technical Skill Gaps" : "Explain the technical skills they're missing, why they're important, and what they can do to develop these skills to close the gaps and qualify for a promotion."
                      "Soft Skills" : "Provide a list of soft skills needed to be successful in this role and provide soft skill development recommendations."
                """
            elif collect_career_goal == "I want to transition into a new role":
                
                # Process the input and generate recommendations
                model_inputs = f"{collect_industry or ''},{top5_skills}"
                role_predictions = predict_top_5_job_roles(model_inputs)

                # Prepare the prompt for Google Gemini
                prompt = f"""You are a career advisor specializing in helping professionals to be promoted in their current role, OR, transition into a new role. You will receive information about a user's current job title, their top 5 skills, their current industry, and a list of target job titles they wish to transition into. Your task is to provide insightful and concise guidance to help them understand the alignment between their skills, predicted roles, and their career goal. For each predicted role, provide a skill match analysis summarizing the *overall* match of the user's skillset to the demands of the role, considering all 5 skills together, and then a detailed breakdown of each skill.

                Here is the user's information:

                Career Goal: {collect_career_goal}
                Current Role: {current_role}
                Current Industry: {collect_industry}
                Top 5 Skills: {(top5_skills)}
                Target Jobs: {(role_predictions)}

                Do not output the response in JSON. Instead, output the response in the structure listed below in the "Roles to Consider" object for all 5 provided target jobs. For the "Skill match analysis explained" you must present the skill match analysis in a table for each of the 5 skills. Don't forget to provide recommendations for developing the skill. If any of the "Target Jobs" are erroneous or not applicable to the top 5 skills provided, please select a job title to replace it from your model without annotating the job was replaced next to the job role. After the 5 jobs, at the very end of the result provide a list of any jobs you replaced.

                  "Roles to consider": 
                      "** Recommendation #: Insert Job Title here ** MAKE THE JOB TITLE 
                      BOLD AND ENSURE IT STANDS OUT FROM THE REST OF THE TEXT **
                      If Gemini replaces a role predicion do not apply any type of label to the job
                      title row.",
                      "Job description": "Concise and informative job description (around 2-3 sentences). Focus on key responsibilities and required skills.",
                      "Average Annual Sallary: "If available, this is as an average annual salary in US dollars for the role. If salary information is unavailable, please indicate it is unavailable.",
                      "Skill match analysis summary": "Overall analytical assessment of how well the user's *entire skillset* (all 5 skills combined) aligns with the requirements of this role. Consider the relative importance of each skill to the role. Examples: 'Strong overall match...', 'Partial match...', 'Weak overall match...'",
                      "Skill match analysis explained":
                          "Skill": "Skill Name",
                          "Match": "Weak/Medium/Strong",
                          "Justification": "Brief explanation."
                      "How your skills apply to this role": "Explain how the user's provided skills can be practically applied to this specific job. Provide concrete examples and focus on transferable skills. Aim for 3-4 sentences."
                      "Skill Development": "Provide recommendations to the user to help them develop this skill."
                
                Introduce a divider line after all 5 predictions share the following information
                with the user:
                      
                `"Jobs Replaced":
                    "Replaced '[role_prediction]' with '[Gemini's recommendation]'"
                    "Replaced '[role_prediction]' with '[Gemini's recommendation]'"
                """
            
            # Call Google Gemini to generate content
            print('Retrieving response from Gemini..')
            with st.spinner("Generating Google Gemini response..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash-002")
                    response = model.generate_content(prompt)

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

