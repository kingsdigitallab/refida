import sqlite3
import tempfile

import numpy as np
from txtai.database import SQLError
from txtai.embeddings.transform import Action
from txtai.pipeline import Segmentation
import settings
from txtai.embeddings import Embeddings, Transform
from refida.data import get_data_path
import re

DUMMYDOCID = 'FIRSTDOC'


class SemIndexDoc:
    """txtai semantic index for the documents"""

    def __init__(self, datadir: str = settings.DATA_DIR.name, session_state=None):
        self.embeddings: Embeddings = None
        if session_state is None:
            session_state = {}
        self.session_state = session_state
        self.datadir = datadir
        self.segmenter = Segmentation(sentences=True)
        self.set_highlight_format()

    def search_docs(self, phrase, limit=None, min_score=settings.SEARCH_MIN_SCORE):
        return self.search_phrase(phrase, limit, min_score)

    def search_phrase(self, phrase, limit=None, min_score=settings.SEARCH_MIN_SCORE):
        # print('SEARCH')
        phrase = self.clean_search_phrase(phrase)

        query = f"select id, text, score, from txtai where similar('{phrase}')"
        if min_score:
            query += f" and score >= {min_score}"
        # tiny text tend to be very similar to any query, e.g. car -> a or 1
        ret = self.search_sql(query, limit=limit)

        return ret

    def clean_search_phrase(self, phrase):
        # drop all boolean operators
        ret = re.sub(r"\b(OR|AND|NOT)\b", "", phrase)
        ret = re.sub(r'"', " ", ret)
        ret = re.sub(r"\s+", " ", ret).strip()
        return ret

    def search_sql(self, query, limit=None):
        if limit is None:
            # in txtai limit=None is the same as limit=3!
            climit = 1000000
        else:
            # see https://github.com/neuml/txtai/issues/199
            climit = limit * 20
        semindex = self.load_embeddings()
        ret = semindex.search(query, limit=climit)
        ret = ret[:limit]
        return ret

    def reindex(self, dataframe, search_column, progressbar=None):

        self.reset_embeddings()

        for r in dataframe.itertuples(False):
            if progressbar:
                progressbar.update(1)
            text = getattr(r, search_column, None)
            if isinstance(text, str) and len(text) > 3:
                # index text
                self.reindex_item(r, text)

        self.embeddings.save(self.get_index_path())

    def reindex_item(self, row, text):
        self.embeddings.upsert([(row.id, {"text": text}, None)])

    def upsert(self, documents):
        """
        Runs an embeddings upsert operation. If the index exists, new data is
        appended to the index, existing data is updated. If the index doesn't exist,
        this method runs a standard index operation.

        Args:
            documents: list of (id, data, tags)
        """
        # ADAPTED from txtai.embeddings.Embeddings.upsert()
        # to return the embedding for documents[0] (as a list).
        #

        # Add dummy doc to create the index
        if not self.embeddings.ann:
            dummy_doc = (DUMMYDOCID, {"text": DUMMYDOCID}, None)
            self.embeddings.upsert([dummy_doc])

        # Create transform action
        transform = Transform(self.embeddings, Action.UPSERT)

        with tempfile.NamedTemporaryFile(mode="wb",
                                         suffix=".npy") as buffer:
            # Load documents into database and transform to vectors
            ids, _, embeddings = transform(documents, buffer)
            if ids:
                # Normalize embeddings
                self.embeddings.normalize(embeddings)

                # Append embeddings to the index
                self.embeddings.ann.append(embeddings)

        return list(embeddings[0])

    def upsert_embedding(self, document, embedding):

        if not self.embeddings.ann:
            dummy_doc = (DUMMYDOCID, {"text": DUMMYDOCID}, None)
            self.embeddings.upsert([dummy_doc])

        # Create transform action
        transform = Transform(self.embeddings, Action.UPSERT)

        with tempfile.NamedTemporaryFile(mode="wb",
                                         suffix=".npy") as buffer:
            # Load documents into database and transform to vectors
            ids, _, embeddings = transform([document], buffer)
            if ids:
                # Normalize embeddings
                embeddings = np.reshape(embedding, [1, embedding.shape[0]])
                self.embeddings.normalize(embeddings)

                # Append embeddings to the index
                self.embeddings.ann.append(embeddings)

    def reset_embeddings(self) -> Embeddings:
        self.embeddings = Embeddings(
            {
                "path": settings.SEARCH_TRANSFORMER,
                # to enable text & metadata storage (i.e. 'documents' sqlite file)
                "content": True,
            }
        )

    def get_index_path(self):
        return str(
            get_data_path(
                self.datadir, "1_interim", "semindex" + self.get_path_suffix()
            )
        )

    def get_path_suffix(self):
        return ""

    def load_embeddings(self) -> Embeddings:

        state_name = "semindex" + self.get_path_suffix()
        ret = self.session_state.get(state_name, None)
        if ret is None:
            ret = Embeddings()
            try:
                # print(f'LOAD EMBDEDDINGS {self.get_index_path()}')
                ret.load(self.get_index_path())
                self.session_state[state_name] = ret
            except FileNotFoundError:
                raise Exception(
                    "The search index is missing."
                    " Run `python cli.py index` to build it."
                )

        self.embeddings = ret

        return ret

    def set_highlight_format(self, before="", after=""):
        self.highlight_before = before
        self.highlight_after = after

    def get_highlighted_text_from_hit(self, hit, phrase, limit=settings.SEARCH_MAX_SNIPPETS):
        ret = hit["text"]

        if not self.highlight_before:
            return ret

        highlights = []

        phrase = self.clean_search_phrase(phrase)

        if settings.SEARCH_EXPLAIN_STRATEGY == 2:
            sents = [s for s in self.segmenter(hit["text"]) if len(s) > 4]
            highlights = [
                [sents[sim[0]], sim[1]]
                for sim in self.load_embeddings().similarity(phrase, sents)
            ]
        if settings.SEARCH_EXPLAIN_STRATEGY == 3:
            index_sents = SemIndexSent(session_state=self.session_state)
            docid = hit["id"]
            try:
                sents = index_sents.search_sql(
                    f"select id, text, docid, score from txtai"
                    f" where docid = '{docid}' and similar('{phrase}')"
                    f" and length(text) > 8",
                    limit=limit,
                )

                highlights = [[sent["text"], sent["score"]] for sent in sents]
            except SQLError:
                message = "(WARNING: search explanation failed)"

        if 0:
            for highlight in highlights:
                ret = ret.replace(
                    highlight[0],
                    self.highlight_before + highlight[0] + self.highlight_after,
                )
        else:
            ret = '<br><br>'.join([
                f'...{highlight[0]}...'
                for highlight
                in highlights[:limit]
            ])

        return ret

    def get_info(self):
        index = self.load_embeddings()
        config = index.config.copy()
        config.pop("ids", None)

        ret = {
            'class': type(self).__name__,
            'filepath': self.get_index_path(),
            'type': 'txtai.Embeddings',
            'config': config,
            'size': index.count(),
        }

        return ret


class SemIndexSent(SemIndexDoc):
    """txtai semantic index for the sentences"""

    def search_docs(self, phrase, limit=None, min_score=settings.SEARCH_MIN_SCORE):
        ret = []
        phrase = self.clean_search_phrase(phrase)
        query = f"select docid, text, score as score from txtai where similar('{phrase}') "
        if min_score:
            query += f" and score >= {min_score}"

        res = self.search_sql(query, limit=limit*1000)
        found = {}
        for r in res:
            if r['docid'] in found:
                continue
            found[r['docid']] = 1
            ret.append({
                'id': r['docid'],
                'text': r['text'],
                'score': r['score']
            })
            if len(ret) >= limit:
                break
        return ret

    def reindex(self, dataframe, search_column, progressbar=None):
        self.sent_idx = -1
        super(SemIndexSent, self).reindex(dataframe, search_column, progressbar)

    def reindex_sentences_and_docs(self, dataframe, search_column, progressbar=None):
        self.sent_idx = -1

        # we index the docs from the sentences
        self.index_doc = SemIndexDoc()
        self.index_doc.reset_embeddings()

        super(SemIndexSent, self).reindex(dataframe, search_column, progressbar)

        # remove dummy doc used to create indices
        self.embeddings.delete([DUMMYDOCID])
        self.embeddings.save(self.get_index_path())
        self.index_doc.embeddings.delete([DUMMYDOCID])

        self.index_doc.embeddings.save(self.index_doc.get_index_path())

    def reindex_item(self, row, text):
        # index sentences
        embeddings = []

        sents = self.segmenter(text)
        for sent in sents:
            self.sent_idx += 1
            embeddings.append(
                self.upsert(
                    [(self.sent_idx, {"text": sent, "docid": row.id}, None)]
                )
            )

        # average embeddings
        doc_embedding = np.average(np.array(embeddings), axis=0)
        self.index_doc.upsert_embedding((row.id, {"text": text}, None), doc_embedding)

    def get_path_suffix(self):
        return "_sents"


class LexicalIndexDoc(SemIndexDoc):
    """sqlite fst5 lexical index (bm25) for the documents"""

    def reindex(self, dataframe, search_column, progressbar=None):
        # con = self.embeddings.database.connection
        con = sqlite3.connect(self.get_index_path())
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS txtsql")
        cur.execute(
            """
            CREATE VIRTUAL TABLE txtsql
            using fts5(text, id, tokenize='porter unicode61')
        """
        )

        idx = 0
        for row in dataframe.itertuples():
            text = getattr(row, search_column, None)
            sql = """INSERT INTO txtsql(text, id) VALUES(?,?)"""
            cur.execute(sql, [text, row.id])
            idx += 1
            if progressbar:
                progressbar.update(1)

        con.commit()
        con.close()

    def get_index_path(self):
        return str(get_data_path(self.datadir, "1_interim", "lexindex"))

    def search_phrase(self, phrase, limit=None, min_score=settings.SEARCH_MIN_SCORE):
        fields = "id, text"
        if self.highlight_before:
            fields += ", snippet(txtsql, 0, '{}', '{}', '...', 64) as highlighted".format(
                self.highlight_before, self.highlight_after
            )


        query = f"""
            select {fields}, 1.0 as score
            from txtsql(?)
        """
        query += " order by rank"

        phrase = self.clean_search_phrase(phrase)

        return self.search_sql(query=query, limit=limit, parameters=[phrase])

    def clean_search_phrase(self, phrase):
        # king's returns Runtime error: fts5: syntax error near "'"
        phrase = phrase.replace("'", ' ')
        return phrase

    def search_sql(self, query, limit=None, parameters=None):
        ret = []

        if limit:
            query += f" limit {limit}"

        if parameters is None:
            parameters = []

        con = sqlite3.connect(self.get_index_path())
        cur = con.cursor()
        # print(query, parameters)
        rows = cur.execute(query, parameters)
        for row in rows:
            hit = {col[0]: row[idx] for idx, col in enumerate(cur.description)}
            hit["score"] = 1.0
            ret.append(hit)
        con.close()

        return ret

    def get_highlighted_text_from_hit(self, hit, phrase):
        ret = hit.get("highlighted", None)
        if ret is None:
            ret = hit["text"]

        return ret

    def get_info(self):
        columns = self.search_sql("pragma table_info('txtsql')")
        counts = self.search_sql("select count(*) as count from txtsql")

        ret = {
            'class': type(self).__name__,
            'filepath': self.get_index_path(),
            'type': 'sqlite3',
            'config': {
                'backend': 'FTS5'
            },
            'schema': {
                'cols': {col['name']: '' for col in columns}
            },
            'size': counts[0]["count"],
        }

        return ret

"""
TODO:
M: index the text (not just summary)
    S: snippets
    S: optimise
S: check caching
C: additional fields
S: fix lexical score (all 1.0...)
"""
