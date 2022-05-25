import sqlite3

from txtai.pipeline import Segmentation
import settings
from txtai.embeddings import Embeddings
from refida.data import get_data_path


class SearchIndexDoc:

    def __init__(self, datadir: str = settings.DATA_DIR.name):
        self.datadir = datadir
        self.segmenter = Segmentation(sentences=True)

    def search(self, phrase):
        ret = []

        return ret

    def reindex(self, dataframe, search_column, progressbar=None):

        # index docs
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


class SearchIndexSent(SearchIndexDoc):

    def reindex(self, dataframe, search_column, progressbar=None):
        self.sent_idx = -1
        super(SearchIndexSent, self).reindex(dataframe, search_column, progressbar)

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


class LexicalIndexDoc(SearchIndexDoc):

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

    def search(self, phrase, limit=None):
        ret = []

        import re
        fields = 'id, text'

        query = f'''
            select {fields}, 1.0 as score
            from txtsql(?)
        '''
        query += ' order by rank'
        if limit is not None:
            query += f' limit {limit}'
        con = sqlite3.connect(self.get_index_path())
        cur = con.cursor()
        rows = cur.execute(query, [phrase])
        for row in rows:
            hit = {
                col: row[idx]
                for idx, col in enumerate(re.findall(r'\w+', fields))
            }
            hit['score'] = 1.0
            ret.append(hit)
        con.close()

        return ret

# TODO:
# M: universal phrase (OR, AND. ...)
# M: snippets for fts5
# M: index the text (not just summary)
# S: optimise
# S: check caching
# C: additional fields
# C: move semsearch to classes
# C rename classes
