import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from typing import List

# Load .env if exists (for LOCAL development)
load_dotenv()

# --- Helper functions -------------------------------------------------------

def build_prompt_template(platform: str, tone: str, description: str, num_variants: int, language: str = "日本語") -> ChatPromptTemplate:
    system_message = f"""
You are an expert copywriter for social media. 
Given a content description, generate {num_variants} unique short captions tailored for {platform} in a {tone} tone.
Each caption should:
- Fit typical length constraints of {platform} (keep it concise)
- Include 1-3 relevant hashtags at the end
- Be distinct from each other
Return the captions as a numbered list (1., 2., ...) with the caption only (no extra explanation).
"""

    human_template = """
Content description:
{description}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )
    return prompt

def parse_captions(raw: str) -> List[str]:
    # Simple split by lines starting with number or dash
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    captions = []
    for line in lines:
        # remove leading "1. " or "2)" etc.
        cleaned = line
        if cleaned[0].isdigit():
            # find first dot or parenthesis
            if "." in cleaned:
                cleaned = cleaned.split(".", 1)[1].strip()
            elif ")" in cleaned:
                cleaned = cleaned.split(")", 1)[1].strip()
        captions.append(cleaned)
    return captions

# --- Streamlit UI ----------------------------------------------------------

st.set_page_config(page_title="SNS キャプションジェネレータ", layout="centered")

st.title("📸 SNS投稿キャプションジェネレータ")
st.markdown(
    """
短い説明から指定のトーン・プラットフォーム向けにキャプション案を複数生成します。  
コピーしてそのまま使える形式で出力。  
"""
)

with st.form("caption_form"):
    description = st.text_area("投稿内容の説明（例：海での週末の写真、コーヒーと読書の時間など）", max_chars=500)
    col1, col2, col3 = st.columns(3)
    with col1:
        platform = st.selectbox("ターゲットSNSプラットフォーム", ["Instagram", "Twitter", "X", "Facebook", "TikTok", "LinkedIn", "Threads"])
    with col2:
        tone = st.selectbox("トーン", ["カジュアル", "エモーショナル", "プロフェッショナル", "ユーモラス", "シンプル", "情熱的"])
    with col3:
        language = st.selectbox("言語", ["日本語", "English"])
    num_variants = st.slider("キャプション案の数", min_value=3, max_value=8, value=5)
    submitted = st.form_submit_button("キャプションを生成")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("環境変数 `OPENAI_API_KEY` が設定されていません。設定してから再実行してください。")

if submitted:
    if not description.strip():
        st.error("説明を入力してください。")
        st.stop()

    with st.spinner("キャプションを生成中..."):
        try:
            # モデルは必要に応じて 'gpt-4o' や 'gpt-4o-mini' などに変えられる
            llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o", max_tokens=400)
            prompt = build_prompt_template(platform=platform, tone=tone, description=description, num_variants=num_variants, language=language)
            chain = LLMChain(llm=llm, prompt=prompt)

            raw_output = chain.run({"description": description})
            captions = parse_captions(raw_output)
            if not captions:
                # フォールバック：生の出力を1つだけ
                captions = [raw_output.strip()]

        except Exception as e:
            st.exception(f"生成中にエラーが発生しました: {e}")
            st.stop()

    st.subheader("キャプション案")
    for idx, cap in enumerate(captions[:num_variants], start=1):
        st.markdown(f"**{idx}.** {cap}")
        st.code(cap, language="text")
        st.markdown("---")

    # コピー用まとめ
    st.subheader("まとめてコピー")
    joined = "\n".join([f"{i}. {c}" for i, c in enumerate(captions[:num_variants], start=1)])
    st.text_area("全案（コピー用）", value=joined, height=200)

    # オプション：文章要約（元説明の短縮版）
    st.subheader("説明の要約（任意）")
    try:
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "あなたは日本語の要約の専門家です。以下の内容説明を読み、読み手に伝わりやすい自然な日本語で、短く一文に要約してください。"
            ),
            HumanMessagePromptTemplate.from_template("{description}")
        ])
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summary_chain.run({"description": description})
        st.info(f"要約: {summary.strip()}")
    except Exception:
        st.info("要約は生成できませんでした。")