from fpdf import FPDF
from io import BytesIO
import time
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

top_skills_response0 = ''
industry_response0 = ''
    
def predict_top_roles(top_skills_response0, industry_response0):
    """
    Predict the top 3 job roles based on the input skills and industry.
    """
    
    # Validate inputs
    if not top_skills_response0:
        top_skills_response0 = "No skills provided"
    if not industry_response0:
        industry_response0 = "No industry provided"
    
    # Combine user inputs into a single feature string
    input_text = top_skills_response0 + ', ' + industry_response0
    
    # Create embedding for the input text
    input_embedding = ml_model.encode([input_text])  # Ensure it's in a list format for batch processing

    # Ensure tensor shape consistency here (example: reshaping if necessary)
    input_embedding = tf.convert_to_tensor(input_embedding, dtype=tf.float32)
    
    # Get predictions from the model
    prediction = loaded_model.predict(input_embedding)  # Probabilities for all roles

    # Get the indices of the top 3 probabilities in descending order
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    
    # Decode the top 3 predicted labels to their corresponding roles
    top_roles = [
        label_encoder.inverse_transform([index])[0]
        for index in top_indices
    ]
    
    return top_roles

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

                    # Add a delay to respect rate limits
                    time.sleep(0.5)
                
                print(results)
                
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

    This is a prototype under development and may contain bugs or errors. It is intended for educational and research purposes only. 
    
    If you test the prototype, please note the following:

    - The app might be rate-limited with sub-optimal performance due to free services and usage limitations.

    - Limits on the number of concurrent users are in effect.

    The job descriptions, salary info, skill match analysis, skill development recommendations, and skill gaps are developed and output from the Google Gemini 2.0 experimental LLM.

    Learning resource links are developed by the Google Search API. 
    
    Skill(X) is powered by the following resources:
    
    - Google Gemini
    - Google Search
    - Kaggle Dataset - https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
    - Modern BERT NLP Model (English)
    - Numpy 
    - Python
    - SKLearn - Label Encoder & Sentence Transformer
    - Streamlit

    ''')

container_x = st.container(border=True)
with container_x:
    st.markdown("""
    <div>
        <h3 style="color:#48acd2; text-align:center;">Plan Your Next Career Move</h3>
        <p style="font-size:17px; line-height:1.6;">
            Skill(X) is fine-tuned and trained on 1.6 million jobs and associated skills to predict roles that match your skills using the pre-trained Modern BERT model.
        </p>
        <p style="font-size:17px; line-height:1.6;">
            Just answer a few simple questions, and we'll provide targeted job recommendations and learning resources.
        </p>
        <p style="font-size:17px; line-height:1.6;">
            Let's begin! 👇 👇
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

container_m = st.container(border=True)

with container_m:
    
    if st.session_state.get("form_submitted", False):

        with st.spinner("Loading model..."):
            print("Loading the trained model file...")
            loaded_model = tf.keras.models.load_model('role_skills_descriptions_industries_trained_model.keras') 
            print("Trained model loaded.")

            # Load the LabelEncoder
            print("Loading LabelEncoder...")
            label_encoder = load('role_skills_descriptions_industries_trained_model.joblib')
            print("LabelEncoder loaded.")

            # Ensure TensorFlow backend is set (if PyTorch is problematic)
            os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "tensorflow"

            ml_model = SentenceTransformer('nomic-ai/modernbert-embed-base', cache_folder='./sbert_cache')

            st.success("Model loaded!", icon="🧠")

if st.session_state.get("form_submitted", False):

    role_predictions = predict_top_roles(top_skills_response0, industry_response0)
    consolidated_prompt = f"""

        Context: You are a highly specialized career advisor with expertise in skill gap analysis and career path development.

        Objective: Your primary goal is to provide users with tailored job role recommendations based on career goal logic and skillset.

        Career Goal Logic:

            If the job seeker's career goal is "I want to be promoted.":
            Recommend 3 roles: a lead or senior role, a management role, and a director or executive role within the user's current industry.

            If the job seeker's career goal is "I want to transition into an entirely new role.":
            Utilize {role_predictions}. However, If any role prediction is entirely unrelated to the job seeker's skills replace the target jobs with a suitable alternative from your own LLM.
            If you replace a target job with a job from your model be sure to annotate the following "This role was selected by Google Gemini"

        User Information and Skillset:

            Career Goal: {response_goal0}
            Current Role: {role_response0}
            Current Industry: {industry_response0}
            Top Skills: {top_skills_response0}
            Target Jobs: {role_predictions}

        Output Requirements:

            Format: Markdown

        Structure:

            For each recommended role:
                "(bold) Recommendation #: " (e.g., "Recommendation #1") followed by **"Job Title" (bold).
                "Job Description" (bold): Concise and informative (2-3 sentences) focusing on key responsibilities and required skills.
                "Average Annual Salary"(bold): If available, provide the average annual salary in US dollars in a uniform font size. If unavailable, state "Salary information unavailable."
                "Skill Match Analysis Summary" (bold): Provide an elaborate analysis describing how the job seeker's top skills and how they align with the recommended role. Classify their skills as a strong, partial, or weak match.
                "Skill Match Analysis Explained" (bold): Present the skill match analysis in a table for all of the top skills provided.
                        "Skill Description"(bold): Provide a description justifying why the provided skill is relevant to the recommended job.
                        "Match Strength" (bold): Classify each skill as either a Strong overall match/partial match/weak overall match based on the skill's relevance to the recommended role.
                            "Strong overall match"
                            "Partial match"
                            "Weak overall match"
                "How Your Skills Apply to This Role"(bold): Explain how the user's provided skills can be practically applied, focusing on transferable skills with 3-4 concrete examples.
                "Skill Development"(bold): Provide up to 5 specific recommendations in a bullet point format for developing any skills with a weak or medium match. 
                "Skill Gaps"(bold): Identify any critical or essential skills for the recommended role which are not provided in the job seeker's top skills: {top_skills_response0}. Provide suggestions to develop the skill gaps you identify.

            Divider: Insert a clear divider (e.g., "---") between each role recommendation.

        Limitations:

            Limit the response to 3 recommended roles.
            Do not include additional language or commentary beyond the specified structure.

        """

response_text = ""
#model = genai.GenerativeModel("gemini-1.5-flash-002")
model = genai.GenerativeModel("gemini-2.0-flash-exp")

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
