import sqlite3

from txtai.database import SQLError
from txtai.pipeline import Segmentation
import settings
from txtai.embeddings import Embeddings
from refida.data import get_data_path
import re

class SemIndexDoc:
    '''txtai semantic index for the documents'''

    def __init__(self, datadir: str = settings.DATA_DIR.name):
        self.datadir = datadir
        self.segmenter = Segmentation(sentences=True)
        self.set_highlight_format()

    def search_phrase(
            self, phrase, limit=None, min_score=settings.SEARCH_MIN_SCORE
    ):
        phrase = self.clean_search_phrase(phrase)

        query = f"select id, text, score, from txtai where similar('{phrase}')"
        if min_score:
            query += f" and score >= {min_score}"
        ret = self.search_sql(query, limit=limit)

        return ret

    def clean_search_phrase(self, phrase):
        # drop all boolean operators
        ret = re.sub(r'\b(OR|AND|NOT)\b', '', phrase)
        ret = re.sub(r'\s+', ' ', ret).strip()
        return ret

    def search_sql(self, query, limit=None):
        semindex = self.read_index()
        return semindex.search(query, limit=limit)

    def reindex(self, dataframe, search_column, progressbar=None):

        self.embeddings = self.new_embeddings()

        for r in dataframe.itertuples(False):
            if progressbar:
                progressbar.update(1)
            text = getattr(r, search_column, None)
            if isinstance(text, str) and len(text) > 3:
                # index text
                self.reindex_doc(self.embeddings, r, text)

        self.embeddings.save(self.get_index_path())

    def reindex_doc(self, embeddings, row, text):
        embeddings.upsert([
            (row.id, {"text": text}, None)
        ])

    def new_embeddings(self) -> Embeddings:
        return Embeddings({
            # "path": "sentence-transformers/nli-mpnet-base-v2",
            # "path": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "path": settings.SEARCH_TRANSFORMER,
            # to enable text & metadata storage (i.e. 'documents' sqlite file)
            "content": True,
        })

    def get_index_path(self):
        return str(get_data_path(
            self.datadir, "1_interim", "semindex" + self.get_path_suffix()
        ))

    def get_path_suffix(self):
        return ""

    def read_index(self) -> Embeddings:
        # TODO: this class should be independent from streamlit
        import streamlit as st
        state_name = "semindex" + self.get_path_suffix()
        ret = st.session_state.get(state_name, None)
        if ret is None:
            ret = Embeddings()
            try:
                ret.load(self.get_index_path())
                st.session_state[state_name] = ret
            except FileNotFoundError:
                st.error(
                    "The search index is missing."
                    " Run `python cli.py reindex` to build it."
                )

        return ret

    def set_highlight_format(self, before='', after=''):
        self.highlight_before = before
        self.highlight_after = after

    def get_highlighted_text_from_hit(self, hit, phrase):
        ret = hit["text"]

        if not self.highlight_before:
            return ret

        highlights = []

        phrase = self.clean_search_phrase(phrase)

        if settings.SEARCH_EXPLAIN_STRATEGY == 2:
            sents = [s for s in self.segmenter(hit["text"]) if len(s) > 4]
            highlights = [
                [sents[sim[0]], sim[1]]
                for sim in self.read_index().similarity(phrase, sents)
            ]
        if settings.SEARCH_EXPLAIN_STRATEGY == 3:
            index_sents = SemIndexSent()
            docid = hit["id"]
            try:
                sents = index_sents.search_sql(
                    f"select id, text, docid, score from txtai "
                    f"where docid = '{docid}' and similar({phrase})",
                    limit=2,
                )

                highlights = [[sent["text"], sent["score"]] for sent in sents]
            except SQLError:
                message = "(WARNING: search explanation failed)"

        for highlight in highlights:
            ret = ret.replace(
                highlight[0],
                self.highlight_before + highlight[0] + self.highlight_after,
            )
            break

        return ret


class SemIndexSent(SemIndexDoc):
    '''txtai semantic index for the sentences'''

    def reindex(self, dataframe, search_column, progressbar=None):
        self.sent_idx = -1
        super(SemIndexSent, self).reindex(dataframe, search_column, progressbar)

    def reindex_doc(self, embeddings, row, text):
        # index sentences
        sents = self.segmenter(text)
        for sent in sents:
            self.sent_idx += 1
            self.embeddings.upsert([
                (self.sent_idx, {"text": sent, "docid": row.id}, None)
            ])

    def get_path_suffix(self):
        return "_sents"


class LexicalIndexDoc(SemIndexDoc):
    '''sqlite fst5 lexical index (bm25) for the documents'''

    def reindex(self, dataframe, search_column, progressbar=None):
        # con = self.embeddings.database.connection
        con = sqlite3.connect(self.get_index_path())
        cur = con.cursor()
        cur.execute('DROP TABLE IF EXISTS txtsql')
        cur.execute('''
            CREATE VIRTUAL TABLE txtsql
            using fts5(text, id, tokenize='porter unicode61')
        ''')

        idx = 0
        for row in dataframe.itertuples():
            text = getattr(row, search_column, None)
            sql = '''INSERT INTO txtsql(text, id)
              VALUES(?,?)'''
            cur.execute(sql, [text, row.id])
            idx += 1

        con.commit()
        con.close()

    def get_index_path(self):
        return str(get_data_path(self.datadir, "1_interim", "lexindex"))

    def search_phrase(
            self, phrase, limit=None,
            min_score=settings.SEARCH_MIN_SCORE
    ):
        fields = "id, text"
        if self.highlight_before:
            fields += ", highlight(txtsql, 0, '{}', '{}') as highlighted".format(
                self.highlight_before,
                self.highlight_after
            )

        query = f'''
            select {fields}, 1.0 as score
            from txtsql(?)
        '''
        query += ' order by rank'

        return self.search_sql(query=query, limit=limit, parameters=[phrase])

    def clean_search_phrase(self, phrase):
        return phrase

    def search_sql(self, query, limit=None, parameters=None):
        ret = []

        if limit:
            query += f' limit {limit}'

        con = sqlite3.connect(self.get_index_path())
        cur = con.cursor()
        rows = cur.execute(query, parameters)
        for row in rows:
            hit = {
                col[0]: row[idx]
                for idx, col in enumerate(cur.description)
            }
            hit['score'] = 1.0
            ret.append(hit)
        con.close()

        return ret

    def get_highlighted_text_from_hit(self, hit, phrase):
        ret = hit.get("highlighted", None)
        if ret is None:
            ret = hit["text"]

        return ret


# TODO:
# M: index the text (not just summary)
    # S: snippets
    # S: optimise
# S: check caching
# C: additional fields
# C: move semsearch to classes
# S: fix lexical score
