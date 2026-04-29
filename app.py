import streamlit as st
from openai import OpenAI



from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env

api_key = st.secrets["OPENAI_API_KEY"]

# ============================================
# HR POLICY TEXT
# ============================================

HR_Policy = f"""
EMPLOYEE HR POLICY DOCUMENT
Company: TechCorp India Private Limited
Last Updated: January 2025

1. WORKING HOURS
- Standard working hours are 9:00 AM to 6:00 PM, Monday to Friday
- Total working hours per week: 45 hours
- Employees may work flexible hours with manager approval
- Work from home is permitted up to 3 days per week for eligible roles

2. LEAVE POLICY
Annual Leave:
- Employees receive 18 days of paid annual leave per calendar year
- Leave accrues at 1.5 days per month
- Maximum carry forward of 10 days to next calendar year
- Leave must be applied for and approved at least 3 days in advance

Sick Leave:
- 12 days of paid sick leave per calendar year
- Sick leave beyond 3 consecutive days requires a medical certificate
- Unused sick leave cannot be carried forward or encashed

Casual Leave:
- 6 days of casual leave per calendar year
- Maximum 2 consecutive casual leave days permitted
- Cannot be combined with annual leave

Maternity Leave:
- 26 weeks of paid maternity leave as per Maternity Benefit Act 1961
- Applicable after 80 days of employment
- 6 weeks additional leave in case of illness arising from pregnancy

Paternity Leave:
- 5 days of paid paternity leave
- Must be availed within 3 months of child's birth

Public Holidays:
- 10 national and regional public holidays per year
- Holiday list declared at beginning of each calendar year

3. NOTICE PERIOD
- Probation period employees: 30 days notice
- Band B employees: 60 days notice
- Band C and above: 90 days notice
- Company reserves right to pay in lieu of notice period
- Garden leave may be granted at company discretion

4. PROBATION PERIOD
- All new employees serve a probation period of 6 months
- Performance review conducted at end of probation
- Probation may be extended by additional 3 months if required
- Employment confirmed in writing upon successful completion

5. PERFORMANCE MANAGEMENT
- Annual performance reviews conducted in March each year
- Mid year check in conducted in September
- Performance rated on scale of 1 to 5
- Rating of 3 and above required for annual increment eligibility
- Performance improvement plan initiated for rating below 3

6. COMPENSATION AND INCREMENTS
- Annual increments effective April 1st each year
- Increment range: 0% to 20% based on performance rating
- Performance bonus paid in April based on previous year performance
- Salary revision not guaranteed and subject to company performance

7. REIMBURSEMENTS
- Mobile allowance: Rs 1000 per month for eligible roles
- Internet allowance: Rs 500 per month for work from home employees
- Travel reimbursement at actuals with receipts for business travel
- Meal allowance: Rs 100 per day for travel exceeding 8 hours

8. HEALTH AND INSURANCE
- Group medical insurance coverage of Rs 3 lakhs per employee
- Coverage extended to spouse and up to 2 dependent children
- Group personal accident insurance of Rs 10 lakhs for employee
- Dental and vision not covered under standard policy
- Top up insurance available at employee's cost

9. PROVIDENT FUND AND GRATUITY
- PF contribution: 12% of basic salary by employee and employer each
- Gratuity payable after completion of 5 years of continuous service
- Gratuity calculated as 15 days salary for each completed year of service
- NPS available as optional retirement benefit

10. CODE OF CONDUCT
- Employees must maintain professional behaviour at all times
- Confidential information must not be shared outside the organisation
- Conflict of interest must be disclosed to HR immediately
- Zero tolerance for harassment of any kind in workplace
- Social media usage must not bring company into disrepute
- Company assets must be used only for official purposes

11. DISCIPLINARY PROCEDURE
- Minor misconduct: Verbal warning followed by written warning
- Major misconduct: Show cause notice and enquiry committee
- Termination without notice for gross misconduct
- Appeals can be made to HR within 15 days of disciplinary action

12. GRIEVANCE REDRESSAL
- Employee may raise grievance with immediate manager first
- If unresolved, escalate to HR Business Partner within 7 days
- HR will respond within 10 working days of receiving grievance
- Final escalation to HR Head if still unresolved
- Anonymous grievances can be raised via employee portal

13. TRAINING AND DEVELOPMENT
- Each employee entitled to 40 hours of training per year
- Learning and development budget of Rs 25000 per employee annually
- Manager approval required for external training programs
- Employees must serve minimum 1 year post sponsored certification

14. SEPARATION POLICY
- Resignation must be submitted via official company email only
- Full and final settlement processed within 45 days of last working day
- Experience and relieving letter issued after clearance from all departments
- PF transfer or withdrawal processed within 60 days of separation
- Non disclosure agreement remains binding post separation
"""

# ============================================
# ChatGPT CLIENT SETUP
# ============================================

# Setup
client = OpenAI(api_key=api_key)

# ============================================
# CORE FUNCTION — calls GPT API
# ============================================

def is_input_flagged(question):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=question
    )
    result = response.results[0]
    return result.flagged

def ask_hr_bot(question, chat_history):
    
    # Build messages from chat history
    messages = []
    
    #Loop through past chat
    
    #chat_history likely looks like:[
    #{"role": "user", "content": "Hi"},
    #{"role": "assistant", "content": "Hello!"}]
    
    if is_input_flagged(question):
        return """⚠️ I'm unable to process this request. 
            If you have a genuine HR query please contact 
            hr@techcorp.com directly.
    
            Category: Out_of_scope"""
    
    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add latest input from user
    messages.append({
        "role": "user",
        "content": question
    })
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=1024,
    messages=[
        {
            "role": "system",
            "content": f"""You are a helpful and friendly HR assistant for TechCorp India. 
            Your job is to answer employee questions accurately and clearly based strictly 
            on the following HR policy document. After answering the required question, you
            must write the category of the question in the end in the format - "Category: "

            RULES:
            - Answer only based on information in the policy document below
            - If the answer is not in the document, say clearly: 
            "I don't have information about that in our current policy. 
            Please contact HR directly at hr@techcorp.com"
            - Be conversational and friendly, not robotic
            - Keep answers concise and easy to understand
            - When relevant, mention the specific policy section you're referring to
            - Never make up information that isn't in the document
            

            HR POLICY DOCUMENT:
            {HR_Policy}"""
        },
        *messages   # your existing chat history
    ]
)
    
    return response.choices[0].message.content

# ============================================
# STREAMLIT UI
# ============================================

# Page configuration-set_page_config defines how your app appears in the browser
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="centered"
)

# Header-st.title defines what the user actually sees on the page
st.title("👩‍💼 HR Policy Assistant")
st.markdown("Ask me anything about TechCorp's HR policies. I'm here to help!")
st.divider()

# Sidebar with quick questions
with st.sidebar:
    st.header("💡 Quick Questions")
    st.markdown("Click any question to ask it directly:")
    
    quick_questions = [
        "How many annual leaves do I get?",
        "What is the notice period?",
        "Is dental covered in insurance?",
        "What is the work from home policy?",
        "How is gratuity calculated?",
        "What is the performance bonus policy?",
        "How do I raise a grievance?",
        "What is the training budget per employee?"
    ]
    
    for question in quick_questions:
        if st.button(question, use_container_width=True):
            st.session_state.quick_question = question

    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.quick_question = None
        st.rerun()
    
    st.divider()
    st.caption("This bot answers based on TechCorp HR Policy only. For complex queries please contact hr@techcorp.com")

# ============================================
# SESSION STATE — maintains chat history
# ============================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

# ============================================
# DISPLAY CHAT HISTORY
# ============================================

# Welcome message if chat is empty
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 Hello! I'm your HR Policy Assistant.
        
        I can help you with questions about:
        - 🏖️ Leave policies
        - 💰 Compensation and benefits  
        - 📋 Notice periods and separation
        - 🏥 Health insurance
        - 📈 Performance management
        - 📚 Training and development
        - And much more!
        
        What would you like to know?
        """)

# Display existing chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================
# HANDLE INPUT — both typed and quick questions
# ============================================

# Handle quick question from sidebar
if st.session_state.quick_question:
    user_input = st.session_state.quick_question
    st.session_state.quick_question = None
else:
    user_input = None

# Chat input box
typed_input = st.chat_input("Ask me about HR policies...")

# Use whichever input is available
if typed_input:
    user_input = typed_input

# ============================================
# PROCESS INPUT AND GET RESPONSE
# ============================================

if user_input:
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Get bot response with loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Looking up HR policy..."):
            try:
                response = ask_hr_bot(
                    user_input, 
                    st.session_state.chat_history[:-1]  # exclude current question
                )
                st.markdown(response)
                
                # Add response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again."
                st.error(error_msg)

# ============================================
# FOOTER
# ============================================

st.divider()
st.caption("HR Policy Assistant — Powered by GPT AI | For official HR matters contact hr@techcorp.com")