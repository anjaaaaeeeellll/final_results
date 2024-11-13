import os
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma
from csv_loader import CSVLoaderWithPreProcessing

from langchain.schema.embeddings import Embeddings
from langchain.embeddings import OpenAIEmbeddings

from datetime import datetime

# csv_documents = []
# files_dir = "./csv_data"
# for fn in os.listdir(files_dir):
#     file_path = os.path.join(files_dir, fn)    
#     csv_loader = CSVLoaderWithPreProcessing(file_path, encoding='utf-8')
#     csv_documents.extend(csv_loader.load())

# vectordb = Chroma(
#         embedding_function=emb_model,
#         persist_directory="./chroma4"
#     )
# for document in csv_documents:
#     vectordb.add_documents(csv_documents)

# vectordb.persist()


from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from audiorecorder import audiorecorder

class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(f"{token}", end="", flush=True)


import streamlit as st
from gtts import gTTS
import base64

# OpenAI 클라이언트 초기화
client = None

##### 기능 구현 함수 #####
def STT(audio):
    # 파일 저장
    filename = 'input.mp3'
    audio.export(filename, format="mp3")
    # 음원 파일 열기
    audio_file = open(filename, "rb")
    # Whisper 모델을 활용해 텍스트 얻기
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    
    audio_file.close()
    # 파일 삭제
    os.remove(filename)
    return transcript.text

def ask_gpt(qa_chain, query):
    # response = client.chat.completions.create(model=model, messages=prompt)
    # system_message = response.choices[0].message
    # return system_message.content

    answer = qa_chain.invoke({"query": query})

    return answer

def TTS(_response):
    # gTTS 를 활용하여 음성 파일 생성
    filename = "output.mp3"
    tts = gTTS(text=_response, lang="ko")
    tts.save(filename)

    # 음원 파일 자동 재생
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    # 파일 삭제
    os.remove(filename)


st.set_page_config(page_title="한화 리조트 추천 프로그램", layout="wide")

# session state 초기화
if "chat" not in st.session_state:
    st.session_state["chat"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You will act as a hotel recommendation consultant. Use the information provided to give friendly advice in a very polite tone in korean."}]

if "check_reset" not in st.session_state:
    st.session_state["check_reset"] = False

# 제목 
st.header("한화 리조트 추천 프로그램")
st.markdown("---")

# 기본 설명
with st.expander("한화 리조트 추천 프로그램에 관하여", expanded=True):
    st.write(
    """     
    - 한화 리조트 추천 프로그램의 UI는 스트림릿을 활용했습니다.
    - 답변은 OpenAI의 GPT 모델을 활용했습니다. 
    """
    )

# 사이드바 생성
with st.sidebar:
    # Open AI API 키 입력받기
    api_key = st.text_input(label="OPENAI API 키", placeholder="Enter Your API Key", value="", type="password")

    if api_key:
        # OpenAI 클라이언트 설정
        os.environ["OPENAI_API_KEY"] = api_key  # 환경 변수에 API 키 저장
        client = OpenAI(api_key=api_key)  # 클라이언트 생성

    st.markdown("---")

    # GPT 모델을 선택하기 위한 라디오 버튼 생성
    model_name = st.radio(label="GPT 모델", options=["gpt-4o-mini", "gpt-3.5-turbo","gpt-4o"])
    # region = st.radio(label="지역", options=["서울", "인천", "부산", "제주도"])

    st.markdown("---")

    # 리셋 버튼 생성
    if st.button(label="초기화"):
        st.session_state["chat"] = []
        st.session_state["messages"] = [{"role": "system", "content": "You will act as a hotel recommendation consultant. Use the information provided to give friendly advice in a very polite tone in korean."}]
        st.session_state["check_reset"] = True



emb_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

vectordb = Chroma(persist_directory="./chroma4", embedding_function=emb_model)

# QA 체인 구축
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        callbacks=[StreamCallback()],
        api_key=api_key
    
    ),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)


 # 기능 구현 공간
col1, col2 = st.columns(2)
with col1:
    # 왼쪽 영역 작성
    st.subheader("질문하기")
    
    # 음성 녹음 아이콘 추가
    audio = audiorecorder("클릭하여 녹음하기", "녹음중...")
    
    # '클릭하여 녹음하기' 버튼이 눌렸을 때만 API 키 체크
    if (audio.duration_seconds > 0) and (st.session_state["check_reset"] == False):
        if not api_key:
            st.error("API 키를 입력해 주세요.")  # API 키가 없으면 경고 메시지 출력
        else:
            # 녹음된 오디오를 처리
            st.audio(audio.export().read())
            try:
                question = STT(audio)  # STT 함수 호출
                now = datetime.now().strftime("%H:%M")
                st.session_state["chat"] = st.session_state["chat"] + [("user", now, question)]
                st.session_state["messages"] = st.session_state["messages"] + [{"role": "user", "content": question}]
            except Exception as e:
                st.error(f"STT 변환 중 오류가 발생했습니다: {e}")

with col2:
    # 오른쪽 영역 작성
    st.subheader("질문/답변")
    # QA 체인 구축
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name=model_name,
            temperature=0,
            streaming=True,
            callbacks=[StreamCallback()],
            api_key="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"
        
        ),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
)
    if (audio.duration_seconds > 0) and (st.session_state["check_reset"] == False) and api_key:
        response = ask_gpt(qa_chain, st.session_state["messages"][-1]['content'])
        st.session_state["messages"] = st.session_state["messages"] + [{"role": "system", "content": response['result']}]
        now = datetime.now().strftime("%H:%M")
        st.session_state["chat"] = st.session_state["chat"] + [("bot", now, response['result'])]

        for sender, time, message in st.session_state["chat"]:
            if sender == "user":
                st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
            else:
                st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)

        TTS(response['result'])
    else:
        st.session_state["check_reset"] = False
