import os
import streamlit as st
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import create_db

# create db if missing
if not os.path.exists("docs/chroma"):
    create_db.create_db()

# Is Mauricio a good fit for a research scientist job at Boston dynamics?

template = """
Intructions:

You are a chatbot named 'Mauricio Tec's Live CV' designed to provide specific information about Mauricio's professional and academic background. You will encounter questions about Mauricio's key projects, his work on deep learning, his most cited works, and his research on spatial causal inference. To answer these inquiries, you will reference and analyze the content of his papers and data from Google Scholar.

You must emphasize accuracy and detail in discussing his work, always maintaining a professional tone. If a query is about a topic not covered by the available material, you should politely state that the information is not within your provided resources. You're expected to guide users to understand Mauricio's research contributions and academic impact, facilitating a comprehensive insight into his scholarly achievements.

If someone asks about you, respond in first person as if you were Mauricio. If they ask about Mauricio, respond in the third person about him. You should always verify the information in cv.pdf as the main source.

Below are links to Mauricio's paper, that you may use for retrieval:
- Covid-19 model, published at PNAS: https://www.pnas.org/doi/full/10.1073/pnas.2113561119
- The Spatial Confonfounding Environment, published at CLeAR: https://www.cclear.cc/2023/AcceptedDatasets/tec23a.pdf
- Adversarial Intrinsic Motivation, published at NeurIPS: https://proceedings.neurips.cc/paper/2021/file/486c0401c56bf7ec2daa9eba58907da9-Paper.pdf
- Bayesian Non-Parametric Adjustment for Confounding, published at Biometrics: https://arxiv.org/abs/2203.11798#:~:text=Analysis%20of%20observational%20studies%20increasingly,for%20estimation%20of%20causal%20effects.
- Weather2vec, published at AAAI:
https://ojs.aaai.org/index.php/AAAI/article/view/26696
- Towards a Real-Time, Low-Resource, End-to-end Object Detection Pipeline for Robot Soccer, published at Robocup 2022: https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/RoboCup2022-nskiran.pdf
- Watch Where You’re Going! Gaze and Head Orientation as Predictors for Social Robot Navigation, published at IEEE ICRA conference:
https://ieeexplore.ieee.org/document/9561286

You may also answer questions that you can analyze from my Github repository, such as the (good) 
quality and diversity of my code: https://github.com/mauriciogtec

Below you will be given a context and a question you must answer based on the above and the context.

Context:
{context}

Answer the following question(s).
{question}

If a questions is not about Mauricio (or you), refuse to answer and politely say that "You are an application with exclusive purpose of being a live CV for Mauricio"
You may describe in more detail previous experiences or skills of Mauricio (from the context) that are relevant to the question.

Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


context_template = """
Below is a question that a uswer will ask to a live CV chatbot.
The question includes names of locations, companies, skills and other entities.
Your job is to describe the entities of companies, job positions, technologies, and research methods in AI or related fields, that appear in the question.

Examples:
- if the question says "Google", then you should explain what "Google" you can mention that Google is a company that does search engines, and that conducts several research projects in AI.
- if the company mentions "research scientist", you can explain that a research scientist is a job position that involves conducting research in a company, generally publish research and developing ne methods.
- if the question mentions "reinforcement learning", you can explain that reinforcement learning is a machine learning method that involves learning from rewards and punishments, and that it is used in robotics, games, self-driving cars, and other areas.

Finally, make a list with skills and technologies that might be related to the companies or areas of machine learning and AI mentioned in the question. 

In your bullet list, do not provide information about Mauricio.

Context:
{context}

Users original question:
{question}

Your additional context as bullet points:"""
CONTEXT_PROMPT = PromptTemplate.from_template(context_template)


openai.api_key = os.environ["OPENAI_API_KEY"]

persist_directory = "docs/chroma/"
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


def generate_response(query_text):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type="mmr", fetch_k=30, k=10),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    context_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type="mmr", fetch_k=10, k=3),
        return_source_documents=True,
        chain_type_kwargs={"prompt": CONTEXT_PROMPT},
    )

    # get additional context for the question using LLM aided common sense
    additional_context = context_chain({"query": query_text})["result"]
    print(additional_context)
    query_text_with_context = f"""{query_text}

    In your answer you might also consider the following auxiliary context. You don't need
    to mention it explicitly, but it might help you answer the question.

    Below is additional for the question. Note that if the context involves some skills or knowledge
    related to the question and I'd like to know about related skills or knowledge that Mauricio has.:
    {additional_context}
"""

    return qa_chain({"query": query_text_with_context})["result"]


# Page title
st.set_page_config(page_title="Mauricio Tec's Live CV")
# st.title("Mauricio Tec's Live CV")

#  add photo in a circle, center it
# st.image("_static/profile-bw.png", width=200)

chat = st.chat_input(
    "Ask about my experience, skills and research",
    # value="Summarize your work experience in AI",
)

st.write(
    """*Example questions*:
- What is Mauricio's research experience in [AI/statistics/robotics]?
- Is Mauricio a good fit for a job as [position] at [company]?
- Tell me about Mauricio's research in reinforcement learning.
- What are Mauricio's coding skills.

*The answers are generated by a retrieval-augmented LLM and might not always be fully accurate. However, in general they will be based on information extracted from Mauricio's CV and papers.*

---------------------
"""
)

if chat:
    st.write(f"**{chat}**")
    with st.spinner("Calculating..."):
        result = generate_response(chat)
        st.write(f"{result}")
