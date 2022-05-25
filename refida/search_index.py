import sqlite3

from txtai.pipeline import Segmentation
import settings
from txtai.embeddings import Embeddings
from refida.data import get_data_path


class SemIndexDoc:
    '''txtai semantic index of the documents'''

    def __init__(self, datadir: str = settings.DATA_DIR.name):
        self.datadir = datadir
        self.segmenter = Segmentation(sentences=True)
        self.set_highlight_pattern()

    def search_phrase(
            self, phrase, limit=None, min_score=settings.SEARCH_MIN_SCORE
    ):
        query = f"select id, text, score, from txtai where similar('{phrase}')"
        if min_score:
            query += f" and score >= {min_score}"
        ret = self.search_sql(query, limit=limit)

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
        # TODO: st should be passed to this fct or constructor
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

    def set_highlight_pattern(self, before='', after=''):
        self.highlight_before = before
        self.highlight_after = after


class SemIndexSent(SemIndexDoc):
    '''txtai semantic index of the sentences'''

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
    '''sqlite fst5 lexical index (bm25) of the documents'''

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


# TODO:
# M: highlight for fts5
# M: universal phrase (OR, AND. ...)
# M: index the text (not just summary)
    # S: snippets
    # S: optimise
# S: check caching
# C: additional fields
# C: move semsearch to classes
# S: dropbox for limit
# S: fix lexical score
