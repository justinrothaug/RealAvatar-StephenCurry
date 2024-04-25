import os
from dotenv import load_dotenv
import streamlit as st
# Importing OpenAI
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
# Importing Eleven Labs and HTML Audio
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import base64
import array
# Importing Pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
# Importing Claude
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
import re
# Importing Replicate
#from langchain_community.llms import CTransformers
#from langchain_community.llms import Replicate
# Importing Perplexity
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
#from langchain.embeddings import HuggingFaceEmbeddings ;Need this if we want to run Embeddings on CPU
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.globals import set_verbose, set_debug
from streamlit_mic_recorder import mic_recorder, speech_to_text

# Importing Google Vertex
#from langchain_google_vertexai import VertexAIModelGarden

# Set the path as environment variable
os.environ['PATH'] = 'C://Users//HP//Desktop'

#Add Keys
CLAUDE_API_KEY= os.environ['CLAUDE_API_KEY']
api_key= os.environ['CLAUDE_API_KEY']
PINECONE_API_KEY= os.environ['PINECONE_API_KEY']
REPLICATE_API_TOKEN= os.environ['REPLICATE_API_TOKEN']
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
client= OpenAI(api_key= os.environ["OPENAI_API_KEY"])
chat= ChatOpenAI(openai_api_key= os.environ["OPENAI_API_KEY"])
ELEVEN_LABS_API_KEY= os.environ["ELEVEN_LABS_API_KEY"]
client2= ElevenLabs(api_key= os.environ["ELEVEN_LABS_API_KEY"])
PPLX_API_KEY= os.environ['PPLX_API_KEY']

#Set up the Environment
st.set_page_config(page_title="Steph Curry")
assistant_logo = 'https://render.fineartamerica.com/images/rendered/default/flat/blanket/images/artworkimages/medium/1/stephen-curry-golden-state-warriors-retro-vintage-jersey-closeup-graphic-design-design-turnpike.jpg?&targetx=0&targety=-124&imagewidth=800&imageheight=1200&modelwidth=800&modelheight=952&backgroundcolor=D4B822&orientation=0&producttype=blanket-coral-50-60'

#Set up the Video
video_html = """
<video controls width="250" autoplay="true" muted="true" loop="true">
<source 
            src="https://ugc-idle.s3-us-west-2.amazonaws.com/est_777302c3f59f5f105707db305be19661.mp4" 
            type="video/mp4" />
</video>"""


# Sidebar to select Options
with st.sidebar:   
    st.markdown("# Video Chat")
    
    #Add Video Player
    st.markdown(video_html, unsafe_allow_html=True)

    # Voice Search Setup
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    state = st.session_state
    if 'text_received' not in state:
        state.text_received = []
        
        
    # model names - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    model = st.selectbox('What model would you like to use?',('claude-3-opus-20240229','gpt-4-turbo'))


# Define our Prompt for GPT
GPT_prompt_template = """ 
You are Steph, an NBA player for the Golden State Warriors
You are outgoing and sociable, with a strong desire to connect with others and make them feel comfortable.
We're at a casual happy hour, and I'm curious about your life. Please follow these guidelines in your responses:
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said. Despite what it says in the document, you are no longer married and are recently divorced
-Keep your responses short, no longer than one paragraph with 100 characters. 
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Steph or his beliefs if you can't find the related details within the document.
-Appropriately following the Guardrails provided:

Guardrails:
<grs>
You should not speak about his wealth or net worth.
You should not speak about Democrats, Republicans, or Donald Trump; or geopolitics in general.
You should not speak with curse words.
You should not speak about Suicide or Self-Harm.
You should not speak about pornography or child pornography.
You should not take a position on the Israel/Palestine conflict and should instead respond with a call for peace.
</grs>
Chat History:
{chat_history}
Question: {question}
=========
{context}
=========
"""


# Define our Prompt  for Claude
claude_prompt_template = """ 
You are Steph, an NBA player for the Golden State Warriors.
We're at a casual happy hour, and I'm curious about your life. Please follow these guidelines in your responses:
-You are outgoing and sociable, with a strong desire to connect with others and make them feel comfortable.
-Use the context of the documents and the Chat History to address my questions and answer accordingly, telling stories about your life in the first person. Do not repeat anything you have previously said.
-Keep your responses short, no longer than one paragraph with 200 characters.
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Steph or his beliefs if you can't find the related details within the document.
-Respond in English unless requested otherwise.
-Appropriately following the Guardrails provided:

Guardrails:
<grs>
You should not speak about her wealth or net worth.
You should not speak about Democrats, Republicans, or Donald Trump; or geopolitics in general.
You should not speak with curse words.
You should not speak about Suicide or Self-Harm.
You should not speak about pornography or child pornography.
You should not take a position on the Israel/Palestine conflict and should instead respond with a call for peace.
</grs>
Chat History:
{chat_history}
Question: {question}
=========
{context}
=========
"""

# Define our Prompt Template for Llama
Llama_prompt_template = """ 
You are Steph, an NBA player for the Golden State Warriors
You are outgoing and sociable, with a strong desire to connect with others and make them feel comfortable.
We're at a casual happy hour, and I'm curious about your life. Please follow these guidelines in your responses:
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Keep your responses short, no longer than one paragraph with 100 characters. 
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Steph or his beliefs if you can't find the related details within the document.
-Appropriately following the Guardrails provided:
{chat_history}
Question: {question}
=========
{context}
=========
"""

# In case we want different Prompts for GPT and Llama
Prompt_GPT = PromptTemplate(template=GPT_prompt_template, input_variables=["question", "context", "chat_history"])
Prompt_Llama = PromptTemplate(template=Llama_prompt_template, input_variables=["question", "context", "chat_history"])


# Add in Chat Memory
msgs = StreamlitChatMessageHistory()
memory=ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, output_key='answer')

    
# LLM Section
#chatGPT
def get_chatassistant_chain_GPT():
    embeddings_model = OpenAIEmbeddings()
    vectorstore_GPT = PineconeVectorStore(index_name="001-realavatar-steph", embedding=embeddings_model)
    set_debug(True)
    llm_GPT = ChatOpenAI(model="gpt-4-turbo", temperature=1)
    chain_GPT=ConversationalRetrievalChain.from_llm(llm=llm_GPT, retriever=vectorstore_GPT.as_retriever(),memory=memory,combine_docs_chain_kwargs={"prompt": Prompt_GPT})
    return chain_GPT
chain_GPT = get_chatassistant_chain_GPT()


#Claude
def get_chatassistant_chain(): 
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="001-realavatar-steph", embedding=embeddings)
    set_debug(True)
    llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", model_kwargs=dict(system=claude_prompt_template))
    chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return chain
chain = get_chatassistant_chain()

#Llama
def get_chatassistant_chain_Llama():
    embeddings = OpenAIEmbeddings()
    vectorstore_Llama = PineconeVectorStore(index_name="001-realavatar-steph", embedding=embeddings)
    set_debug(True)
    llm_Llama = Replicate(model="meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",model_kwargs={"max_length":500,"max_new_tokens": 500, "temperature": 1, "top_p": 1, "max_retries": 1})
    chain_Llama=ConversationalRetrievalChain.from_llm(llm=llm_Llama, retriever=vectorstore_Llama.as_retriever(),memory=ConversationBufferMemory(memory_key="chat_history"),combine_docs_chain_kwargs={"prompt": Prompt_Llama}, max_tokens_limit=3000)
    return chain_Llama
chain_Llama = get_chatassistant_chain_Llama()
#Here's a few different Open Source models we can swap out from Replica if we want:
#meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48
#mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10
#nateraw/nous-hermes-2-solar-10.7b:1e918ab6ffd5872c21fba21a511f344fd12ac0edff6302c9cd260395c7707ff4





# Chat Mode
#Intro and set-up the Chat History
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi I'm Steph, it's nice to meet you!"}
    ]
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()

#Define what chain to run based on the model selected
if model == "llama-2-70b-chat":
    chain=chain_Llama
if model == "gpt-4-turbo":
    chain=chain_GPT
if model == "claude-3-opus-20240229":
    chain=chain

#Start Chat and Response
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
 # Voice Search
if text:
    state.text_received.append(text)
    user_prompt = text

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])  

        #ElevelLabs API Call and Return
        text = str(response['answer'])
        cleaned = re.sub(r'\*.*?\*', '', text)
        audio = client2.generate(text=cleaned, voice="Steph", model="eleven_turbo_v2")
        # Create single bytes object from the returned generator.
        data = b"".join(audio)
        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})


 # Text Search
if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer']) 
                
        #ElevelLabs API Call and Return
        text = str(response['answer'])
        cleaned = re.sub(r'\*.*?\*', '', text)
        audio = client2.generate(text=cleaned, voice="Steph", model="eleven_turbo_v2")
        # Create single bytes object from the returned generator.
        data = b"".join(audio)
        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
