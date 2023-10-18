import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import GuardrailsOutputParser

os.environ['OPENAI_API_KEY'] = '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'

chat_model = ChatOpenAI()

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

memory = ConversationBufferMemory(memory_key='chat_history', k=5)

prompt = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template="""You are a kind agent, you help humans with code generation tasks
    chat history: {chat_history}
    Human: {question}
    AI:""")

llmchain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

st.set_page_config(
    page_title='Code Generator UI',
    page_icon='..',
    layout='wide')
st.title('Code Generator UI')

total_tokens = 0
amount_spent = 0

col1, col2 = st.columns(2)

with col1:
    programming_languages = ["Python", "JavaScript", "Java", "C++", "Others"]
    chosen_language = st.selectbox("Choose a Programming Language:", programming_languages)
    
    user_prompt = st.text_area("Enter your prompt here:", "")
    
    if st.button('Submit'):
        with get_openai_callback() as cb:
            ai_code_response = llmchain.predict(question=user_prompt)
            ai_sample_input = llmchain.predict(question='Give me sample input, make it simple, it should not be lengthy, do not write output here. Side headings ot text are not needed, simply write the input')
            ai_sample_output = llmchain.predict(question='Give me sample output, make it simple, it should not be lengthy, do not write input here. Side headings ot text are not needed, simply write the output')
            
            st.session_state.ai_code_response = ai_code_response
            st.session_state.ai_sample_input = ai_sample_input
            st.session_state.ai_sample_output = ai_sample_output

            total_tokens = cb.total_tokens
            amount_spent = cb.total_cost
            
            st.write("Sample Input:")
            st.text(st.session_state.ai_sample_input)
            
            st.write("Sample Output:")
            st.text(st.session_state.ai_sample_output)

with col2:
    if 'ai_code_response' in st.session_state:
        st.write(f"Number of tokens used: {total_tokens}") 
        st.write(f"Amount spent in USD: ${amount_spent}") 
        st.write("Generated Code:")
        st.code(st.session_state.ai_code_response, language=chosen_language.lower())
