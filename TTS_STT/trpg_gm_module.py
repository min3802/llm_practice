import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# OpenAI 클라이언트 초기화 (API 키를 입력하세요)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_trpg_world(user_input):
    system_prompt = (
        "당신은 'TRPG'의 숙련된 게임 마스터입니다. "
        "플레이어의 말이나 행동을 바탕으로 현재 상황에 숨겨진 비밀이나 풍부한 세계관 설정을 상세히 작성하세요. "
        "판타지 소설처럼 몰입감 있는 한국어로 작성하세요."
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"플레이어의 선언: {user_input}"}
        ],
        temperature=0.8,
        max_tokens=800
    )
    
    return response.choices[0].message.content

def summarize_for_player(master_lore):
    system_prompt = (
        "당신은 TRPG 마스터입니다. 제시된 마스터의 비밀 설정을 바탕으로 플레이어에게 직접 말해줄 진행 대사를 작성하세요. "
        "규칙: 1. 한국어로만 말할 것. 2. 3문장 이내로 요약할 것. 3. 마지막은 반드시 질문으로 끝낼 것."
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"비밀 설정 내용: {master_lore}"}
        ],
        temperature=0.5,
        max_tokens=300
    )
    
    return response.choices[0].message.content

print("✅ [TRPG GM 모듈] 준비 완료!")