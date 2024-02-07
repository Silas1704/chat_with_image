import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectdetectionTool
import os

background_css = """
        body {
            background-color: #f0f0f0; 
            
             background-image: url('https://deci.ai/wp-content/uploads/2023/08/deci-langchain-featured.jpg'); 

        }
    """

st.markdown(f'<style>{background_css}</style>', unsafe_allow_html=True)


tools=[ImageCaptionTool(),ObjectdetectionTool()]
conversational_memory=ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm=ChatOpenAI(
    openai_api_key='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',
    temperature=0.8,
    model_name="gpt-3.5-turbo"

)


agent=initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)




st.title('Ask a question to an image')
st.header('Upload Image')
file=st.file_uploader("Choose an image....",type=["jpeg","jpg","png"])
if file:
    # display image
    st.image(file,use_column_width=True)
    user_question=st.text_input('Ask a question about your image')
    with NamedTemporaryFile(dir='D:\Chat_with_image',delete=False) as f:
        f.write(file.getbuffer())
        image_path = f.name

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
                st.write(response)
