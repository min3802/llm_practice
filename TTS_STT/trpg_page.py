import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
import time
from faster_whisper import WhisperModel
from gtts import gTTS
from trpg_gm_module import create_trpg_world, summarize_for_player

# --- 1. 설정 및 모델 로드 ---
st.set_page_config(page_title="노닥노닥 TRPG", page_icon="🎲")

@st.cache_resource
def load_stt_model():
    return WhisperModel("base", device="cuda", compute_type="float16")

stt_model = load_stt_model()

# --- 2. 세션 상태 초기화 (기억 장치) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "당신은 TRPG 마스터입니다."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # (role, text, audio_path) 저장
if "is_world_set" not in st.session_state:
    st.session_state.is_world_set = False
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None # 무한 루프 방지의 핵심!

# --- 3. 사이드바 (초기화 및 종료) ---
with st.sidebar:
    if st.button("🔄 새 게임 시작"):
        # 모든 파일 삭제 및 세션 초기화
        for f in os.listdir():
            if f.startswith("voice_") or f == "temp_input.wav":
                os.remove(f)
        st.session_state.clear()
        st.rerun()

# --- 4. 메인 화면 및 대화창 ---
st.title("🎲 노닥노닥 TRPG")

# 대화 기록 출력 (플레이 버튼 포함)
for role, text, audio_path in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path) # 답변 바로 아래에 재생 버튼 생성

# --- 5. 입력 및 처리 로직 (중요!) ---
st.write("---")
label = "🎤 배경 설정하기" if not st.session_state.is_world_set else "⚔️ 다음 행동 선언"
audio_input = mic_recorder(start_prompt=label, stop_prompt="🛑 녹음 중단", key='recorder')

# [무한 루프 방지 관문]
# 1. 녹음 데이터가 있어야 함
# 2. 그 데이터의 ID(해시값)가 이전에 처리한 것과 달라야 함
if audio_input:
    current_audio_id = hash(audio_input['bytes'])
    
    if current_audio_id != st.session_state.last_audio_id:
        # 즉시 ID를 업데이트하여 다음 재실행 때 통과 못 하게 막음
        st.session_state.last_audio_id = current_audio_id
        
        with st.spinner("마스터가 듣고 있습니다..."):
            # 소리 저장 및 STT
            with open("temp_input.wav", "wb") as f:
                f.write(audio_input['bytes'])
            segments, _ = stt_model.transcribe("temp_input.wav", language="ko")
            user_text = "".join([s.text for s in segments])

        if user_text.strip():
            # 플레이어 입력 기록
            st.session_state.chat_history.append(("user", user_text, None))
            
            # 맥락에 따른 입력 구성
            if not st.session_state.is_world_set:
                input_msg = f"[세계관 설정]: {user_text}\n이 배경으로 첫 상황을 시작해줘."
                st.session_state.is_world_set = True
            else:
                input_msg = f"[플레이어 행동]: {user_text}"
            
            st.session_state.messages.append({"role": "user", "content": input_msg})

            # 마스터 답변 생성
            with st.spinner("마스터가 생각 중..."):
                master_lore = create_trpg_world(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": master_lore})
                
                player_summary = summarize_for_player(master_lore)
                
                # TTS 파일 생성
                tts_file = f"voice_{int(time.time())}.mp3"
                gTTS(text=player_summary, lang='ko').save(tts_file)
                
                # 마스터 답변 기록 (텍스트와 오디오 경로 함께 저장)
                st.session_state.chat_history.append(("assistant", player_summary, tts_file))
            
            # 모든 처리가 끝났으므로 화면을 새로고침해서 대화창에 반영
            st.rerun()