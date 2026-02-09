# app.py
import os
import json
import time
from datetime import datetime, timedelta

import requests
import streamlit as st

# OpenAI SDK (new style)
from openai import OpenAI


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI 습관 트래커",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관 + 기분 + 날씨 + 강아지로 AI 코치 리포트를 만들어봐요 🐶")


# =========================
# Constants
# =========================
HABITS = [
    ("기상 미션", "⏰"),
    ("물 마시기", "💧"),
    ("공부/독서", "📚"),
    ("운동하기", "🏃"),
    ("수면", "😴"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Suwon",
    "Ulsan",
    "Jeju",
    "Sejong",
]

COACH_STYLES = {
    "스파르타 코치": "sparta",
    "따뜻한 멘토": "mentor",
    "게임 마스터": "gm",
}


# =========================
# Helpers: API
# =========================
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap에서 현재 날씨를 가져옵니다.
    - 한국어
    - 섭씨
    - 실패 시 None
    - timeout=10
    """
    try:
        if not api_key:
            return None

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }

        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        # 안전하게 파싱
        weather_desc = None
        if data.get("weather") and isinstance(data["weather"], list) and len(data["weather"]) > 0:
            weather_desc = data["weather"][0].get("description")

        main = data.get("main", {})
        wind = data.get("wind", {})

        result = {
            "city": data.get("name", city),
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "weather": weather_desc,
            "wind_mps": wind.get("speed"),
        }

        # 필수값 없으면 None 처리
        if result["temp_c"] is None and result["weather"] is None:
            return None

        return result

    except Exception:
        return None


def get_dog_image():
    """
    Dog CEO API에서 랜덤 강아지 이미지 URL과 품종을 가져옵니다.
    - 실패 시 None
    - timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        if data.get("status") != "success":
            return None

        image_url = data.get("message")
        if not image_url:
            return None

        # 품종 파싱: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
        breed = "Unknown"
        try:
            parts = image_url.split("/breeds/")[1].split("/")
            breed_raw = parts[0]  # e.g. "hound-afghan"
            breed = breed_raw.replace("-", " ").title()
        except Exception:
            breed = "Unknown"

        return {"image_url": image_url, "breed": breed}

    except Exception:
        return None


# =========================
# AI Report
# =========================
def _system_prompt(style_key: str) -> str:
    if style_key == "sparta":
        return (
            "너는 엄격하고 단호한 '스파르타 코치'다.\n"
            "- 핑계 금지, 행동 중심.\n"
            "- 짧고 강하게 말한다.\n"
            "- 비난이 아니라 훈련/피드백의 톤.\n"
            "- 오늘의 성과를 냉정하게 평가하고 내일 미션을 명확히 제시한다."
        )
    if style_key == "mentor":
        return (
            "너는 따뜻하고 다정한 '멘토'다.\n"
            "- 사용자를 응원하고 감정을 존중한다.\n"
            "- 작은 성취도 인정해준다.\n"
            "- 현실적인 조언을 부드럽게 제안한다.\n"
            "- 말투는 편안하고 친근하다."
        )
    # gm
    return (
        "너는 RPG 세계관의 '게임 마스터'다.\n"
        "- 사용자의 하루를 퀘스트/스탯/레벨업처럼 묘사한다.\n"
        "- 재미있고 몰입감 있게 말한다.\n"
        "- 하지만 조언은 실제로 도움이 되게 구체적으로 준다."
    )


def generate_report(
    openai_api_key: str,
    coach_style_key: str,
    today_data: dict,
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달.
    모델: gpt-5-mini
    실패 시 None 반환
    """
    try:
        if not openai_api_key:
            return None

        client = OpenAI(api_key=openai_api_key)

        # 습관 요약
        checked = today_data.get("checked", {})
        mood = today_data.get("mood", None)
        city = today_data.get("city", None)
        achievement_rate = today_data.get("achievement_rate", None)

        habit_lines = []
        for habit_name, emoji in HABITS:
            val = bool(checked.get(habit_name, False))
            habit_lines.append(f"- {emoji} {habit_name}: {'완료' if val else '미완료'}")

        # 날씨 요약
        if weather:
            weather_line = (
                f"{weather.get('city', city)} / {weather.get('weather')} / "
                f"{weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C) / "
                f"습도 {weather.get('humidity')}%"
            )
        else:
            weather_line = "날씨 정보를 가져오지 못함"

        # 강아지 요약
        if dog:
            dog_line = f"{dog.get('breed', 'Unknown')}"
        else:
            dog_line = "강아지 정보 없음"

        system = _system_prompt(coach_style_key)

        # 출력 형식 강제
        format_rules = """
출력은 반드시 아래 형식을 지켜라. (마크다운 OK)

[컨디션 등급] S/A/B/C/D 중 하나
[습관 분석] (핵심 3줄 + 가장 중요한 1개 습관을 지정)
[날씨 코멘트] (날씨가 없으면 그에 맞게)
[내일 미션] 3개 (체크박스 습관과 연결)
[오늘의 한마디] 1~2문장

추가 규칙:
- 과장 금지, 현실적인 조언
- 사용자의 기분(1~10)을 반드시 반영
- 달성률(%)을 반드시 반영
"""

        user = f"""
사용자 오늘 체크인 데이터:

도시: {city}
기분(1~10): {mood}
달성률(%): {achievement_rate}

습관 체크:
{chr(10).join(habit_lines)}

날씨:
{weather_line}

오늘의 랜덤 강아지 품종:
{dog_line}

위 정보를 바탕으로 리포트를 작성해줘.
{format_rules}
"""

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )

        text = resp.choices[0].message.content
        return text

    except Exception:
        return None


# =========================
# Session State: demo data + today
# =========================
def _init_demo_data():
    """
    6일 샘플 데이터 + 오늘(비어있는 상태) 준비.
    session_state에 저장.
    """
    if "history" in st.session_state:
        return

    today = datetime.now().date()
    # 6일 전 ~ 1일 전: 샘플
    demo = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        # 샘플 패턴(랜덤 없이 고정)
        checks = {
            "기상 미션": i % 2 == 0,
            "물 마시기": True,
            "공부/독서": i % 3 != 0,
            "운동하기": i % 2 != 0,
            "수면": True if i % 4 != 0 else False,
        }
        checked_count = sum(1 for v in checks.values() if v)
        rate = int(round((checked_count / len(HABITS)) * 100))
        mood = max(1, min(10, 6 + (2 - (i % 5))))

        demo.append(
            {
                "date": d.isoformat(),
                "checked_count": checked_count,
                "achievement_rate": rate,
                "mood": mood,
            }
        )

    st.session_state.history = demo

    # 오늘 데이터(기본값)
    st.session_state.today_checked = {name: False for name, _ in HABITS}
    st.session_state.today_mood = 6
    st.session_state.today_city = "Seoul"
    st.session_state.coach_style = "스파르타 코치"


_init_demo_data()


# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔑 API 설정")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="예: sk-... (환경변수 OPENAI_API_KEY도 사용 가능)",
    )

    weather_api_key = st.text_input(
        "OpenWeatherMap API Key",
        type="password",
        value=os.environ.get("OPENWEATHER_API_KEY", ""),
        help="OpenWeatherMap에서 발급받은 API Key",
    )

    st.divider()
    st.caption("키는 브라우저 세션에서만 사용되며, 서버에 저장하지 않습니다.")


# =========================
# Main Layout
# =========================
left, right = st.columns([1.05, 1.0], gap="large")


# =========================
# Left: Check-in UI
# =========================
with left:
    st.subheader("✅ 오늘 체크인")

    # 체크박스 5개를 2열 배치
    col1, col2 = st.columns(2, gap="small")

    # 2열에 적당히 분배: 3 / 2
    for idx, (habit_name, emoji) in enumerate(HABITS):
        target_col = col1 if idx in [0, 1, 2] else col2
        with target_col:
            st.session_state.today_checked[habit_name] = st.checkbox(
                f"{emoji} {habit_name}",
                value=st.session_state.today_checked.get(habit_name, False),
                key=f"habit_{habit_name}",
            )

    st.write("")

    st.session_state.today_mood = st.slider(
        "🙂 오늘 기분은 어때요?",
        min_value=1,
        max_value=10,
        value=int(st.session_state.today_mood),
        help="1=최악, 10=최고",
    )

    city_col, style_col = st.columns([1, 1], gap="medium")

    with city_col:
        st.session_state.today_city = st.selectbox(
            "🌍 도시 선택",
            options=CITIES,
            index=CITIES.index(st.session_state.today_city)
            if st.session_state.today_city in CITIES
            else 0,
        )

    with style_col:
        st.session_state.coach_style = st.radio(
            "🧠 코치 스타일",
            options=list(COACH_STYLES.keys()),
            index=list(COACH_STYLES.keys()).index(st.session_state.coach_style)
            if st.session_state.coach_style in COACH_STYLES
            else 0,
            horizontal=False,
        )

    # 달성률 계산
    checked_count = sum(1 for v in st.session_state.today_checked.values() if v)
    achievement_rate = int(round((checked_count / len(HABITS)) * 100))

    # Metric 3개
    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        st.metric("달성률", f"{achievement_rate}%")
    with m2:
        st.metric("달성 습관", f"{checked_count} / {len(HABITS)}")
    with m3:
        st.metric("기분", f"{st.session_state.today_mood} / 10")

    st.divider()

    # 기록 저장 (session_state)
    save_col, info_col = st.columns([0.55, 0.45], gap="medium")

    with save_col:
        if st.button("💾 오늘 기록 저장", use_container_width=True):
            today = datetime.now().date().isoformat()

            # history에 오늘이 이미 있으면 업데이트, 없으면 추가
            updated = False
            for row in st.session_state.history:
                if row["date"] == today:
                    row["checked_count"] = checked_count
                    row["achievement_rate"] = achievement_rate
                    row["mood"] = st.session_state.today_mood
                    updated = True
                    break

            if not updated:
                st.session_state.history.append(
                    {
                        "date": today,
                        "checked_count": checked_count,
                        "achievement_rate": achievement_rate,
                        "mood": st.session_state.today_mood,
                    }
                )

            # 최근 7개만 유지(데모 + 오늘)
            st.session_state.history = st.session_state.history[-7:]
            st.success("오늘 기록을 저장했어요!")

    with info_col:
        st.caption("※ 저장은 이 브라우저 세션에서만 유지돼요.")


# =========================
# Right: Chart + Report
# =========================
with right:
    st.subheader("📈 7일 달성률 차트")

    # 6일 샘플 + 오늘 데이터 포함해서 7일 만들기
    today_iso = datetime.now().date().isoformat()
    history = list(st.session_state.history)

    # 오늘이 history에 없으면, 임시로 오늘 데이터를 붙여서 차트에만 반영
    if not any(r["date"] == today_iso for r in history):
        history.append(
            {
                "date": today_iso,
                "checked_count": checked_count,
                "achievement_rate": achievement_rate,
                "mood": st.session_state.today_mood,
            }
        )

    # 7개 보장(데모가 6개라서)
    history = history[-7:]

    # 차트 데이터 구성
    labels = []
    values = []
    for r in history:
        d = datetime.fromisoformat(r["date"]).strftime("%m/%d")
        labels.append(d)
        values.append(r["achievement_rate"])

    chart_data = {"date": labels, "achievement_rate": values}
    st.bar_chart(chart_data, x="date", y="achievement_rate")

    st.divider()

    st.subheader("🧾 AI 코치 컨디션 리포트")

    # 결과 저장용 state
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_weather" not in st.session_state:
        st.session_state.last_weather = None
    if "last_dog" not in st.session_state:
        st.session_state.last_dog = None
    if "share_text" not in st.session_state:
        st.session_state.share_text = None

    # 생성 버튼
    if st.button("✨ 컨디션 리포트 생성", type="primary", use_container_width=True):
        with st.spinner("날씨/강아지/AI 코치를 소환 중... 🧙‍♂️"):
            # API 호출
            weather = get_weather(st.session_state.today_city, weather_api_key)
            dog = get_dog_image()

            today_payload = {
                "date": today_iso,
                "checked": st.session_state.today_checked,
                "mood": st.session_state.today_mood,
                "city": st.session_state.today_city,
                "checked_count": checked_count,
                "achievement_rate": achievement_rate,
            }

            style_key = COACH_STYLES.get(st.session_state.coach_style, "sparta")

            report = generate_report(
                openai_api_key=openai_api_key,
                coach_style_key=style_key,
                today_data=today_payload,
                weather=weather,
                dog=dog,
            )

            st.session_state.last_weather = weather
            st.session_state.last_dog = dog
            st.session_state.last_report = report

            # 공유용 텍스트 만들기
            weather_short = "날씨 정보 없음"
            if weather:
                weather_short = f"{weather.get('weather')} / {weather.get('temp_c')}°C"

            dog_short = "🐶 없음"
            if dog:
                dog_short = f"🐶 {dog.get('breed', 'Unknown')}"

            share = f"""AI 습관 트래커 체크인 🧾

📅 날짜: {today_iso}
🌍 도시: {st.session_state.today_city}
🙂 기분: {st.session_state.today_mood}/10
✅ 달성률: {achievement_rate}% ({checked_count}/{len(HABITS)})

오늘 습관:
- ⏰ 기상 미션: {"완료" if st.session_state.today_checked["기상 미션"] else "미완료"}
- 💧 물 마시기: {"완료" if st.session_state.today_checked["물 마시기"] else "미완료"}
- 📚 공부/독서: {"완료" if st.session_state.today_checked["공부/독서"] else "미완료"}
- 🏃 운동하기: {"완료" if st.session_state.today_checked["운동하기"] else "미완료"}
- 😴 수면: {"완료" if st.session_state.today_checked["수면"] else "미완료"}

🌦️ 오늘 날씨: {weather_short}
{dog_short}

🧠 코치 스타일: {st.session_state.coach_style}
"""
            st.session_state.share_text = share

        if st.session_state.last_report is None:
            st.error("리포트 생성에 실패했어요. API Key 또는 네트워크를 확인해줘요 🙏")

    # =========================
    # Result Display
    # =========================
    if st.session_state.last_report:
        weather = st.session_state.last_weather
        dog = st.session_state.last_dog
        report = st.session_state.last_report

        wcol, dcol = st.columns(2, gap="medium")

        # Weather Card
        with wcol:
            st.markdown("### 🌦️ 오늘 날씨")
            if weather:
                st.info(
                    f"**{weather.get('city', st.session_state.today_city)}**\n\n"
                    f"- 상태: {weather.get('weather')}\n"
                    f"- 기온: {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)\n"
                    f"- 습도: {weather.get('humidity')}%\n"
                    f"- 바람: {weather.get('wind_mps')} m/s"
                )
            else:
                st.warning("날씨 정보를 가져오지 못했어요. (OpenWeatherMap 키/도시/네트워크 확인)")

        # Dog Card
        with dcol:
            st.markdown("### 🐶 오늘의 강아지")
            if dog:
                st.caption(f"품종: **{dog.get('breed', 'Unknown')}**")
                st.image(dog["image_url"], use_container_width=True)
            else:
                st.warning("강아지 이미지를 가져오지 못했어요. (Dog CEO API 실패)")

        st.markdown("### 🧠 AI 코치 리포트")
        st.markdown(report)

        st.markdown("### 📌 공유용 텍스트")
        if st.session_state.share_text:
            st.code(st.session_state.share_text, language="text")

    st.divider()

    # =========================
    # API 안내
    # =========================
    with st.expander("ℹ️ API 안내 / 문제 해결", expanded=False):
        st.markdown(
            """
**1) OpenAI API Key**
- OpenAI 플랫폼에서 발급한 키를 입력하세요.
- 모델은 `gpt-5-mini`를 사용합니다.
- 키가 없으면 리포트 생성이 실패합니다.

**2) OpenWeatherMap API Key**
- https://openweathermap.org/ 에서 가입 후 API Key를 발급받아 입력하세요.
- 도시를 영어로 선택합니다(Seoul, Busan 등).
- 무료 플랜은 호출 제한이 있을 수 있어요.

**3) Dog CEO API**
- 키 없이 사용 가능한 무료 API입니다.
- 간헐적으로 실패할 수 있으며, 실패 시 None 처리합니다.

**4) 저장**
- 이 앱은 `st.session_state` 기반이라 브라우저 새로고침/재실행 시 기록이 초기화됩니다.
- 원하면 CSV/DB 저장 기능도 추가해줄게요.
"""
        )
