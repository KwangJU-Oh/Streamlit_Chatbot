import streamlit as st
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# ChatLLM 클래스 정의
class ChatLLM:
    def __init__(self):
        # 모델 설정
        self._model = ChatOllama(model="gemma2:2b", temperature=3)
        
        # 템플릿 설정
        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요.
                            Question: {question}"""
        self._prompt = ChatPromptTemplate.from_template(self._template)
        
        # Chain 연결 (질문 -> 템플릿 -> 모델 -> 출력 파서)
        self._chain = LLMChain(prompt=self._prompt, llm=self._model, output_parser=StrOutputParser())

    def invoke(self, user_input):
        # 사용자 입력을 체인에 전달하고 응답을 반환
        response = self._chain.run({"question": user_input})
        return response


# ChatWeb 클래스 정의
class ChatWeb:
    def __init__(self, llm, page_title="Gazzi Chatbot", page_icon=":books:"):
        self._llm = llm
        self._page_title = page_title
        self._page_icon = page_icon

    def print_messages(self):
        # 세션에 저장된 메시지 출력
        if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
            for chat_message in st.session_state["messages"]:
                st.chat_message(chat_message["role"]).write(chat_message["content"])

    def run(self):
        # 웹 페이지 기본 설정
        st.set_page_config(page_title=self._page_title, page_icon=self._page_icon)
        st.title(self._page_title)

        # 대화 기록 목록을 초기화
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # 이전 대화 기록 출력
        self.print_messages()

        # 사용자 입력 및 AI 응답 처리
        if user_input := st.chat_input("질문을 입력해주세요."):
            # 사용자 입력 처리
            st.chat_message("user").write(f"{user_input}")
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # 모델 응답 처리
            response = self._llm.invoke(user_input)
            with st.chat_message("assistant"):
                st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})


# 실행 코드
if __name__ == '__main__':
    llm = ChatLLM()  # LLM 객체 생성
    web = ChatWeb(llm=llm)  # Web 객체 생성
    web.run()  # 웹 애플리케이션 실행
