import operator
import streamlit as st
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ==========================================
# 1. LangGraph 구조 및 에이전트 정의
# ==========================================

class TRPGState(TypedDict):
    progression: int       # 턴마다 10씩 증가, 100에서 엔딩
    current_story: Annotated[List[str], operator.add] 
    player_input: str
    is_valid_action: bool
    refusal_reason: str

llm = init_chat_model('gpt-4o-mini') 

class ValidatorOutput(BaseModel):
    is_valid_action: bool = Field(description="행동이 물리적/논리적/상황적으로 가능한지 여부")
    refusal_reason: str = Field(description="거절 시 사유 (통과 시 빈 문자열)")

def state_manager_node(state: TRPGState):
    """스탯이 없어졌으므로, LLM 호출 없이 진행도만 10 증가시킵니다."""
    new_progression = state['progression'] + 10 
    return {"progression": new_progression}

def validator_node(state: TRPGState):
    """스탯 없이 순수하게 맥락과 상식 선에서 개연성을 판단합니다."""
    validator_llm = llm.with_structured_output(ValidatorOutput)
    
    # 누적된 전체 스토리를 하나의 텍스트로 합침
    history_text = "\n".join(state['current_story'])
    
    prompt = f"""당신은 TRPG 개연성 판정관입니다.
    [지금까지의 이야기 흐름]
    {history_text}
    
    - 플레이어의 이번 행동: {state['player_input']}
    
    위 이야기 흐름(세계관, 현재 상황)에 비추어 볼 때 플레이어의 행동이 상식적으로 가능한지 판정하세요. 
    터무니없는 행동이거나 세계관의 물리법칙/설정을 무시하는 행동이라면 거절하세요."""
    
    result = validator_llm.invoke(prompt)
    return {
        "is_valid_action": result.is_valid_action, 
        "refusal_reason": result.refusal_reason
    }

def storyteller_node(state: TRPGState):
    """진행도에 따라 스토리를 전개하며, 100에서는 결말을 냅니다."""
    prog = state['progression']
    
    # 누적된 전체 스토리를 하나의 텍스트로 합침
    history_text = "\n".join(state['current_story'])
    
    if prog == 10: 
        instruction = "게임의 첫 시작입니다. 사용자의 입력을 핵심 모티브로 삼아, 이 TRPG의 전체적인 세계관과 매력적인 배경을 설정해 주세요."
    elif prog <= 40:
        instruction = "모험의 초반부입니다. 서브퀘스트나 갈등의 단초를 던져주세요."
    elif prog <= 70:
        instruction = "스토리의 중반부입니다. 결말을 향해 달려가는 긴박한 전개를 작성하세요."
    elif prog <= 90:
        instruction = "클라이막스입니다. 최종 시련을 묘사하세요."
    else: 
        instruction = "모험의 결말입니다. 지금까지 플레이어가 해온 행동과 선택의 맥락을 종합하여, 그에 걸맞은 해피엔딩 또는 배드엔딩 중 하나로 대단원의 막을 내리세요."

    if prog < 100:
        choice_instruction = """
        [선택지 작성 규칙]
        플레이어의 자율성을 극대화하기 위해, 서로 완전히 다른 접근 방식의 선택지 3개를 제시하세요.
        1. 정석적/직접적인 접근 (무력, 정면 돌파 등)
        2. 우회적/은밀한 접근 (잠입, 속임수, 교섭 등)
        3. 환경을 활용하거나 창의적/엉뚱한 접근
        
        그리고 선택지 밑에 반드시 이 문장을 추가하세요: 
        "* 💡 물론 위 선택지에 얽매이지 않고, 원하시는 행동을 자유롭게 입력하셔도 됩니다!"
        """
    else:
        choice_instruction = ""

    prompt = f"""당신은 TRPG 마스터입니다.
    [지금까지의 이야기]
    {history_text}
    
    - 진행도: {prog}/100
    - 플레이어 최신 행동: {state['player_input']}
    
    위 이야기의 맥락과 설정을 완벽하게 유지하면서 다음 이야기를 이어가세요.
    지시사항: {instruction} {choice_instruction}"""
    
    response = llm.invoke(prompt)
    return {"current_story": [response.content]}

def check_validity(state: TRPGState):
    if state["is_valid_action"]:
        return "valid"
    return "invalid"

workflow = StateGraph(TRPGState)
workflow.add_node("validator", validator_node)
workflow.add_node("state_manager", state_manager_node)
workflow.add_node("storyteller", storyteller_node)

workflow.set_entry_point("validator")
workflow.add_conditional_edges("validator", check_validity, {"valid": "state_manager", "invalid": END})
workflow.add_edge("state_manager", "storyteller")
workflow.add_edge("storyteller", END)

trpg_app = workflow.compile()

# ==========================================
# 2. Streamlit 웹 UI 실행부
# ==========================================

st.set_page_config(page_title="LLM TRPG", page_icon="📖", layout="wide")
st.title("📖 10턴의 이야기")

# 초기 상태 세션에 저장
if "trpg_state" not in st.session_state:
    st.session_state.trpg_state = {
        "progression": 0, 
        "current_story": ["환영합니다. 어떤 세계관에서 모험을 시작하고 싶은지, 첫 번째 설정을 입력해 주세요. (예: 마법과 기계가 공존하는 스팀펑크 세계의 뒷골목)"], 
        "player_input": "", 
        "is_valid_action": True, 
        "refusal_reason": ""
    }

state = st.session_state.trpg_state

# --- 왼쪽 사이드바: 진행 상태 ---
with st.sidebar:
    st.header("⏳ 진행 상황")
    st.progress(min(state['progression'], 100) / 100, text=f"진행도: {state['progression']}/100 턴")
    st.divider()
    st.write("턴당 진행도가 10씩 오르며, 100이 되면 플레이어의 행적에 따라 엔딩이 결정됩니다.")
    # 🔄 처음부터 시작하기 버튼 추가
    if st.button("🔄 처음부터 다시 시작", use_container_width=True):
        # 세션 상태(저장된 게임 기록)를 완전히 비움
        st.session_state.clear()
        # 화면을 강제로 새로고침하여 초기 상태로 되돌림
        st.rerun()


# --- 메인 화면: 스토리 로그 ---
if not state["is_valid_action"]:
    st.error(f"❌ 행동 불가: {state['refusal_reason']}")
    st.info("상황에 맞는 다른 행동을 입력해 주세요.")

for idx, msg in enumerate(state["current_story"]):
    with st.chat_message("assistant" if idx == 0 or idx % 2 != 0 else "user"):
        st.write(msg)

# --- 하단: 플레이어 입력창 ---
if state["progression"] >= 100:
    st.success("🎉 이야기가 결말을 맺었습니다. 다시 플레이하려면 새로고침 해주세요!")
else:
    user_input = st.chat_input("당신의 행동이나 설정을 입력하세요...")
    
    if user_input:
        # 화면에 유저 입력 표시를 위해 임시로 current_story에 추가
        state["current_story"].append(user_input)
        state["player_input"] = user_input
        
        with st.spinner("마스터가 이야기를 이어가고 있습니다... ✍️"):
            new_state = trpg_app.invoke(state)
            st.session_state.trpg_state = new_state
            st.rerun()