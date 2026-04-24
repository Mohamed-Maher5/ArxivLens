import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv('ARXIVLENS_API_URL', 'http://localhost:8000').rstrip('/')
INGEST_TIMEOUT_SECONDS = int(os.getenv("ARXIVLENS_INGEST_TIMEOUT", "900"))

st.set_page_config(page_title='ArxivLens', page_icon='📚', layout='wide')

if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ''


def _get(path: str, **params: Any) -> Any:
    response = httpx.get(
        f'{API_BASE_URL}{path}',
        params=params or None,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _post(path: str, payload: dict[str, Any]) -> Any:
    timeout = INGEST_TIMEOUT_SECONDS if path == '/ingest' else 120
    response = httpx.post(
        f'{API_BASE_URL}{path}',
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


st.title('ArxivLens')
st.caption('Search papers, select one, and chat with its Qdrant-backed chunks. No history is persisted.')
st.info(
    "How to use this page: 1) search for a paper or show indexed papers, "
    "2) click Select on one paper, 3) chat with that paper, or 4) ingest a new arXiv ID into Qdrant."
)

query = st.text_input('Search arXiv', placeholder='e.g. vision transformer')
col1, col2 = st.columns([1, 1])
with col1:
    search_clicked = st.button('Search arXiv', use_container_width=True)
with col2:
    refresh_clicked = st.button('Show Indexed Papers In Qdrant', use_container_width=True)

if search_clicked and query.strip():
    try:
        st.session_state.search_results = _get('/papers', q=query.strip(), max_results=8)
        st.session_state.last_query = query.strip()
    except Exception as error:
        st.error(f'Search failed: {error}')
elif refresh_clicked:
    try:
        st.session_state.search_results = _get('/papers')
        st.session_state.last_query = ''
    except Exception as error:
        st.error(f'Loading indexed papers failed: {error}')

papers: list[dict[str, Any]] = st.session_state.search_results
if papers:
    if st.session_state.last_query:
        st.subheader(f"Search Results for '{st.session_state.last_query}'")
    else:
        st.subheader('Indexed Papers')
    for paper in papers:
        title = paper.get('title', paper['arxiv_id'])
        authors = ', '.join(paper.get('authors', [])[:3]) or 'Unknown authors'
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(f"`{paper['arxiv_id']}`")
            st.write(authors)
            st.caption("Click Select to make this the active paper for chat.")
            if st.button(f"Select This Paper", key=f"select-{paper['arxiv_id']}"):
                st.session_state.selected_paper = paper
                st.session_state.chat_messages = []
                st.rerun()
elif search_clicked or refresh_clicked:
    st.warning('No papers found.')

st.divider()
st.subheader('Ingest Paper')
st.caption('Use this when a paper is not indexed yet. It fetches the paper and stores its vectors in local Qdrant.')
st.caption('The first ingest can take several minutes while the PDF is downloaded, parsed, and the embedding model warms up.')
ingest_id = st.text_input('ArXiv ID', placeholder='e.g. 1706.03762')
if st.button('Ingest This ArXiv ID Into Qdrant', use_container_width=True):
    if not ingest_id.strip():
        st.warning('Enter an arXiv ID first.')
    else:
        with st.spinner('Ingesting paper into local Qdrant. This may take a few minutes on first run...'):
            try:
                result = _post('/ingest', {'arxiv_id': ingest_id.strip()})
                st.success(f"Indexed {result['arxiv_id']} with {result['chunk_count']} chunks.")
                st.session_state.selected_paper = result.get('paper')
                st.session_state.chat_messages = []
            except Exception as error:
                st.error(f'Ingest failed: {error}')

selected = st.session_state.selected_paper
if selected:
    st.divider()
    st.subheader('Chat')
    st.write(f"Selected paper: **{selected.get('title', selected['arxiv_id'])}** (`{selected['arxiv_id']}`)")
    st.caption('Each message is independent. The backend sees only this prompt plus fresh retrieval from Qdrant.')
    if st.button('Clear Current Chat', use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message.get('sources'):
                st.caption('Sources: ' + ' | '.join(
                    f"{source.get('paper_title', 'Unknown')} p.{source.get('page_number', '?')}"
                    for source in message['sources'][:3]
                ))

    prompt = st.chat_input('Ask about the selected paper')
    if prompt:
        st.session_state.chat_messages = []
        st.session_state.chat_messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            with st.spinner('Searching Qdrant and generating answer...'):
                try:
                    result = _post('/chat', {'message': prompt, 'arxiv_id': selected['arxiv_id']})
                    st.markdown(result['answer'])
                    if result.get('sources'):
                        st.caption('Sources: ' + ' | '.join(
                            f"{source.get('paper_title', 'Unknown')} p.{source.get('page_number', '?')}"
                            for source in result['sources'][:3]
                        ))
                    st.session_state.chat_messages.append(
                        {
                            'role': 'assistant',
                            'content': result['answer'],
                            'sources': result.get('sources', []),
                        }
                    )
                except Exception as error:
                    st.error(f'Chat failed: {error}')

else:
    st.info('No paper selected yet. Start by searching, showing indexed papers, or ingesting an arXiv ID.')
