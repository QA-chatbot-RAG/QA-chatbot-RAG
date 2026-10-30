import os

from dotenv import load_dotenv
import requests
import streamlit as st


load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="건강정보 챗봇", page_icon="💬")
st.title("건강정보 챗봇")

with st.sidebar:
    st.header("설정")
    top_k = st.slider("검색 문서 수 (top_k)", min_value=1, max_value=20, value=5)
    st.caption(f"백엔드: {BACKEND_URL}")
    st.subheader("최근 응답 시간")
    timings = st.session_state.get("last_latency")
    if timings:
        rerank = "사용 안 함" if timings["rerank"] == 0 else f"{timings['rerank']}"
        st.caption(
            f"embed {timings['embed']} / 검색 {timings['search']} / "
            f"rerank {rerank} / 생성 {timings['generation']} / "
            f"전체 {timings['total']} (ms)"
        )
    else:
        st.caption("아직 응답이 없습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption("※ 위키피디아 기반 일반 정보이며 의학적 진단이나 조언이 아닙니다.")
            if message.get("no_answer"):
                st.info("문서에서 답을 찾지 못했습니다. 다른 질문으로 시도해 주세요.")
            if message.get("sources"):
                with st.expander(
                    f"근거 문서 ({len(message['sources'])})", expanded=False
                ):
                    for source in message["sources"]:
                        title = f"[{source['title']}] " if source.get("title") else ""
                        st.markdown(
                            f"**{title}{source['doc_id']}** · 점수: {source['score']:.4f}"
                        )
                        st.write(source["text"])
                        st.divider()


for item in st.session_state.messages:
    render_message(item)

loading_placeholder = st.empty()
question = st.chat_input("질문을 입력하세요")
if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        with loading_placeholder.container():
            with st.spinner("답변 생성 중..."):
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": question, "top_k": top_k},
                    timeout=90,
                )
                if response.status_code >= 400:
                    try:
                        detail = response.json().get("error", {}).get(
                            "message", response.text
                        )
                    except ValueError:
                        detail = response.text
                    raise RuntimeError(
                        f"백엔드 오류 ({response.status_code}): {detail}"
                    )
                data = response.json()
                assistant_message = {
                    "role": "assistant",
                    "content": data["answer"],
                    "no_answer": data["no_answer"],
                    "sources": data["sources"],
                }
        st.session_state.last_latency = data["latency_ms"]
        st.session_state.messages.append(assistant_message)
        st.rerun()
    except requests.RequestException as exc:
        assistant_message = {
            "role": "assistant",
            "content": f"백엔드에 연결할 수 없습니다. 백엔드 상태를 확인해 주세요. ({exc})",
            "sources": [],
        }
    except (RuntimeError, KeyError, ValueError) as exc:
        assistant_message = {
            "role": "assistant",
            "content": f"요청을 처리하지 못했습니다. {exc}",
            "sources": [],
        }
    finally:
        loading_placeholder.empty()

    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)
