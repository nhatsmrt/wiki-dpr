from wiki_passage_retriever.index import index_wikipedia, retrieve_by_index
import os


class TestIndexing:
    def test_index(self):
        os.mkdir("tmpindex")

        try:
            index_wikipedia("Nelson Mandela", "index")
            candidates = retrieve_by_index("index", "Who was Nelson Mandela's father", 5)

            assert len(candidates) == 5
            # check that the retriever finds the result:
            assert list(filter(lambda pas: "Gadla Henry Mphakanyiswa Mandela" in pas, candidates))
        finally:
            os.remove("tmpindex")
