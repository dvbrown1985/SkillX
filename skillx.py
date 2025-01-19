from fpdf import FPDF
from io import BytesIO
import os
import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from joblib import load
from PIL import Image
from googlesearch import search
from tensorflow.keras import Input
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
    
if "response_text" not in st.session_state:
    st.session_state.response_text = ""

def initialize_session_state():
    default_state = {
        "messages": [],
        "form_submitted": False,
        "response_text": "",
        "state": {
            "current_step": 1,
            "career_goal": None,
            "response_goal": None,
            "current_industry": None,
            "industry_response": None,
            "current_role": None,
            "role_response": None,
            "top_skills_input": None,
            "top_skills_response": None,
            "recommendation": None,
        },
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()    
    
def generate_ai_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-002")
        response = model.generate_content(prompt)
        result = "".join(chunk.candidates[0].content.parts[0].text for chunk in response)
        return result
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def predict_top_3_job_roles(model_inputs):
    """
    Predicts the top 3 most probable job roles based on the given skills.

    Args:
    skills_text: A string containing the skills of the individual.

    Returns:
    A list of tuples, where each tuple contains the predicted job role and its probability.
    """
    print("Loading the trained model file...")
    loaded_model = tf.keras.models.load_model('role_skills_descriptions_industries_trained_model.keras') 
    print("Trained model loaded.")

    # Load the LabelEncoder
    print("Loading LabelEncoder...")
    label_encoder = load('role_skills_descriptions_industries_trained_model.joblib')
    print("LabelEncoder loaded.")

    # Ensure TensorFlow backend is set (if PyTorch is problematic)
    os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "tensorflow"
    
    print("Loading ML model for inference...")
    try:
        ml_model = SentenceTransformer('all-mpnet-base-v2', cache_folder='./sbert_cache')
        print("ML model loaded.")
    except Exception as e:
        print(f"Error loading ML model: {e}")
    
    print('Creating skill embeddings...')
    # Preprocess the skills text
    skill_embedding = ml_model.encode(model_inputs)
    input_data = tf.convert_to_tensor(skill_embedding, dtype=tf.float32)
    input_data = tf.expand_dims(input_data, axis=0)

    predictions = loaded_model.predict(input_data)
    top_3_indices = np.argsort(predictions[0])[::-1][:3]
    top_3_roles = label_encoder.inverse_transform(top_3_indices)
    top_3_probabilities = predictions[0][top_3_indices]

    predicted_roles = [(role, prob) for role, prob in zip(top_3_roles, top_3_probabilities)]
    print(predicted_roles)
    st.success('Predictions generated!', icon="🤖")
    return predicted_roles

def extract_job_titles(response_text):
    
# Display a spinner with the message while the operation is in progress
    with st.spinner("Locating free learning resources..."):

        # Extract job titles from the response_text using a regular expression
        job_titles = re.findall(r'Recommendation\s+#\d+:\s+(.+)', response_text)

        # Remove any asterisks (*) from the job titles
        job_titles = [title.replace('*', '') for title in job_titles]

        # Ensure we only process the top 3 job titles, if there are at least 3
        if len(job_titles) >= 3:
            job_titles = job_titles[:3]  # Slice the list to include only the first 3 titles

            # Iterate over the selected job titles with an index (starting at 1)
            for i, job_title in enumerate(job_titles, 1):

                # Create a search query to find free learning resources for the job title
                learning_plan = f"I'm looking for free courses, tutorials, and podcasts to develop skills to become a {job_title}"

                # Initialize an empty list to store search results
                results = []

                # Perform a search for the learning plan and iterate through the results
                for j, result in enumerate(search(learning_plan)):
                    results.append(result)  # Add the result to the list

                    # Limit the number of results to 3
                    if j >= 2:
                        break

                # Append a message to the session state with links for the current job title
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Learning options for {job_title}\n" + "\n".join([
                        # Format each link with HTML for a clickable link with styling
                        f'<a href="{link}" target="_blank" style="color: #1f77b4; text-decoration: none; overflow: hidden; text-overflow: ellipsis; display: block; max-width: 100%; margin-bottom: 5px;">{link}</a>'
                        for link in results
                    ])
                })
        else:
            # Raise an error if fewer than 3 job titles were found
            raise ValueError("Less than 3 job titles were found in the response.")

        st.success('Learning resources found!', icon="🤖")

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

container1 = st.container(border=True)
with container1:
    st.image(image, caption=None)

# Disclaimer
expander = st.expander("Disclaimer and Product Information")
expander.write('''
    This is a prototype under development and may contain bugs or errors. 
    It is intended for testing and educational purposes only. 
    Please use this prototype with caution and at your own risk.
    
    Skill(X) is powered by the Modern BERT model, Google Gemini, Google Search, Numpy, Python, Sentence Transformer, SKLearn, Streamlit and TensorFlow.

''')

container_x = st.container(border=True)
with container_x:
    st.markdown("""
    <div> 
        <h3 style="color:#48acd2; text-align:center;">Discover your next career move</h3>
        <p style="font-size:17px; line-height:1.6;">
            Skill(X) analyzes 1.5 million jobs and skills to find your perfect role. In the event your skillset doesn't exist in the Skill(X) training data, Google Gemini selects roles for you.
        </p>
        <p style="font-size:17px; line-height:1.6;">
            After answering a few questions, Skill(X) provides detailed job recommendations with descriptions, salary ranges, skill assessments, and links to free training.
        </p>
        <p style="font-size:17px; line-height:1.6;">
            Let's get started 👇 👇
        </p>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.state["current_step"] == 1:
    st.chat_message("assistant").write("Are you looking for a promotion? Or, do you want to transition into a new role?")
    career_goal_input = st.chat_input("Enter your career goal")

    if career_goal_input:
        goal_prompt = f"""
        Analyze the following chatbot user response: '{career_goal_input}'.
        Determine the user's primary career aspiration and assess whether they want a promotion, or if they want to transition into a new role:
         - Promotion: The user desires advancement within their current field or company.
         - Career Change: The user seeks a completely different job function.
        You may only reply with two options:
        I want to be promoted.
        I want to transition into an entirely new role.
        """

        with st.spinner("Analyzing..."):
            response_goal = generate_ai_response(goal_prompt)
            print(response_goal)
            if career_goal_input:
                st.session_state.state["career_goal"] = career_goal_input
                st.session_state.state["response_goal"] = response_goal
                st.session_state.state["current_step"] = 2
                st.rerun()

elif st.session_state.state["current_step"] == 2:
    st.chat_message("assistant").write("What industry do you currently work in?")
    industry_input = st.chat_input("Enter your industry")
    industry_prompt = f"""
    1. Context: "You are analyzing user responses from a chatbot. Specifically, you are tasked with identifying the user's current industry of employment."
   "The user will provide a single-sentence response to the question: 'In what industry do you currently work?'"
    2. User Response: "{industry_input}"
    3. Instructions: "Analyze the user's response and identify the primary industry. Provide your answer in a single word or a short, concise phrase (e.g., 'Technology', 'Healthcare', 'Manufacturing', 'Finance'). If the user's response is ambiguous or does not clearly indicate an industry, respond with 'Indeterminate'."
    4. Example: 
   User Response: "I work for a company that develops and sells software for businesses."
   Your Response: "Technology"
    """    

    with st.spinner("Analyzing..."):
        response_industry = generate_ai_response(industry_prompt)
        if industry_input:
            st.session_state.state["current_industry"] = industry_input
            st.session_state.state["industry_response"] = response_industry
            st.session_state.state["current_step"] = 3
            st.rerun()

elif st.session_state.state["current_step"] == 3:
    st.chat_message("assistant").write("What is your current role?")
    role_input = st.chat_input("Enter your role")
    role_prompt = f"""
    You are assisting a user with correcting and formatting job titles and MUST follow the instructions below:
        1. User input: "{role_input}"
        2. Task: Correct any spelling errors in the user’s input and return a concise job title in title case (e.g., "Solution Consultant", "Software Engineer", "Data Scientist"). 
        3. Example:
            User input: "I am a soluton consltnt"
            Your response: "Solution Consultant"
    """

    with st.spinner("Analyzing..."):
        response_role = generate_ai_response(role_prompt)
        if role_input:
            st.session_state.state["current_role"] = role_input
            st.session_state.state["role_response"] = response_role
            st.session_state.state["current_step"] = 4
            st.rerun()

elif st.session_state.state["current_step"] == 4:
    st.chat_message("assistant").write("What are your top skills?")
    top_skills_input = st.chat_input("Enter your top skills")
    skills_prompt = f"""
    1. Context: "You are analyzing a user's response to the question: 'What are your top skills?'" "The user's response may contain a list of skills presented in various formats (e.g., full sentences, phrases, bullet points)." 2. User Response: "{top_skills_input}" 3. Instructions: 1. Extract Skills: Identify and extract individual skills from the user's response. Skills may be expressed as verbs (e.g., "developing", "managing"), noun phrases (e.g., "python programming", "API implementation"), or other grammatical structures. 2. Clean and Standardize: Remove irrelevant words and phrases like (e.g., "I", "am", "proficient", "in", "and", "the"). Remove phrases like "I can", "I have experience in", "I am skilled in". Standardize formatting: Convert all skills to lowercase. Remove any extra spaces or punctuation. 3. Output: Present the extracted skills as a single line, separated by commas. Each skill should be presented as a single, concise term or short phrase. Example Output: Demonstrations, Proof-of-concept management, Solution design, Python programming, API implementation.
    """

    with st.spinner("Analyzing..."):
        response_skills = generate_ai_response(skills_prompt)
        if top_skills_input:
            st.session_state.state["top_skills_input"] = top_skills_input
            st.session_state.state["top_skills_response"] = response_skills
            st.session_state.state["current_step"] = 5
            st.session_state.state["recommendation"] = 'ready'
            st.rerun()

elif st.session_state.state["current_step"] == 5:
    with st.form("user_profile"):
        st.info(f"""
        **Your Skill(X) profile:**
        \n   🏅  **Career Goal:** {st.session_state.state['response_goal']}
        \n   ℹ️  **Current Industry:** {st.session_state.state['industry_response']}
        \n   ⚙️  **Current Role:** {st.session_state.state['role_response']}
        \n   🥷  **Top Skills:** {st.session_state.state['top_skills_response']}
        """
        )
        
        #st.info(f"- **Career Goal:** {st.session_state.state['response_goal']}")
        #st.info(f"- **Current Industry:** {st.session_state.state['industry_response']}")
        #st.info(f"- **Current Role:** {st.session_state.state['role_response']}")
        #st.info(f"- **Top Skills:** {st.session_state.state['top_skills_response']}")

        if not st.session_state.form_submitted:
            submitted = st.form_submit_button("Generate recommendations")

        if submitted:
            st.session_state.form_submitted = True
            
response_goal0 = st.session_state.state['response_goal']
print(response_goal0)
industry_response0 = st.session_state.state['industry_response']
print(industry_response0)
role_response0 = st.session_state.state['role_response']
print(role_response0)
top_skills_response0 = st.session_state.state['top_skills_response']
print(top_skills_response0)

role_predictions = ""
    
if response_goal0 == 'I want to transition into an entirely new role.':
        
    if st.session_state.get("form_submitted", False):
        model_inputs = f"{industry_response0}, {top_skills_response0}"
        st.session_state["role_predictions"] = predict_top_3_job_roles(model_inputs)
        role_predictions = st.session_state["role_predictions"]
    else:
        print(f" Role repsonse: {role_response0}")
        
consolidated_prompt = f"""

    Context: You are a highly specialized career advisor with expertise in skill gap analysis and career path development.

    Objective: Your primary goal is to provide users with tailored career recommendations based on career goal logic and skillset.

    Career Goal Logic:

        If the job seeker's career goal is "I want to be promoted.":
        Recommend 3 roles: a lead or senior role, a management role, and a director or executive role within the user's current industry.
        
        If the job seeker's career goal is "I want to transition into an entirely new role.":
        Utilize {role_predictions} as a starting point.
        
        If any role prediction is deemed inappropriate, replace it with a suitable alternative.
        Recommend 3 suitable roles based on the user's skills and career goals.
    
    User Information:

    Career Goal: {response_goal0}
    Current Role: {role_response0}
    Current Industry: {industry_response0}
    Top Skills: {top_skills_response0}
    Target Jobs: {role_predictions}
    
    Output Requirements:

        Format: Markdown

    Structure:
        
        For each recommended role:
            "Recommendation #: (bold)" (e.g., "Recommendation #1") followed by **"Job Title" (bold). If a role prediction is deemed inappropriate, replace it with a suitable alternative from your knowledge base without any annotation.
            "Job Description" (bold and prominent): Concise and informative (2-3 sentences) focusing on key responsibilities and required skills.
            "Average Annual Salary"(bold font and prominent): If available, provide the average annual salary in US dollars in a uniform font size. If unavailable, state "Salary information unavailable."
            "Skill Match Analysis Summary" (bold and prominent): Provide an elaborate analysis describing how the job seeker's top skills and how they align with the recommended role. Classify their skills as a strong, partial, or weak match.
            "Skill Match Analysis Explained" (bold and prominent): Present the skill match analysis in a table for all of the top skills provided.
                    "Skill Description"(bold and prominent): Provide a description justifying why the provided skill is relevant to the recommended job.
                    "Match Strength" (bold and prominent): Classify each skill as either a Strong overall match/partial match/weak overall match based on the skill's relevance to the recommended role.
                        "Strong overall match"
                        "Partial match"
                        "Weak overall match"
            "How Your Skills Apply to This Role"(bold and prominent): Explain how the user's provided skills can be practically applied, focusing on transferable skills with 3-4 concrete examples.
            "Skill Development"(bold and prominent): Provide up to 5 specific recommendations in a bullet point format for developing any skills with a weak or medium match. 
            "Skill Gaps"(bold and prominent): Identify any critical or essential skills for the recommended role which are not provided in the job seeker's top skills: {top_skills_response0}. Provide suggestions to develop the skill gaps you identify.
    
        Divider: Insert a clear divider (e.g., "---") between each role recommendation.
    
    Limitations:

        Limit the response to 3 recommended roles.
        Do not include additional language or commentary beyond the specified structure.
    
    """

response_text = ""
model = genai.GenerativeModel("gemini-1.5-flash-002")

@st.fragment
def download_recommendations():
    # Collect all messages in the format 'role: content'
    all_messages_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
    
    # Provide a download button for the generated PDF
    st.download_button(
        label="Download learning resources & recommendations",
        data=all_messages_text,
        file_name="SkillX_recommendations.txt",
        mime="text/plain"
    )

if st.session_state.get("form_submitted", False):
    
    print(consolidated_prompt)
    with st.spinner("Developing recommendations..."):
        try:
            # Validate that a career goal exists and is supported
            if consolidated_prompt:  # Check if the consolidated_prompt is not empty
                # Generate content using the model
                response = model.generate_content(consolidated_prompt)
                response_text = ""
                for chunk in response:
                    text_content = chunk.candidates[0].content.parts[0].text
                    response_text += text_content

                # Append the generated response to the session messages
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.error("Prompt was empty. Check your logic.")  # Error message if the prompt is empty
        except (KeyError, IndexError) as e:
            st.error(f"Error extracting chunk text: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

container2 = st.container(border=True)
with container2:
    if st.session_state.get("form_submitted", False):
        st.success("Recommendations developed!", icon="🤖")
        extract_job_titles(response_text)
        download_recommendations()
    
container3 = st.container(border=True)
if st.session_state.messages:
    for message in st.session_state.messages:
        with container3.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
