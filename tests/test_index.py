from wiki_passage_retriever.index import index_wikipedia, retrieve_by_index
import shutil


class TestIndexing:
    def test_index(self):
        try:
            index_wikipedia("Nelson Mandela", "tmpindex")
            candidates = retrieve_by_index("tmpindex", "Who was Nelson Mandela's father", 5)

            assert len(candidates) == 5
            # check that the retriever finds the result:
            assert list(filter(lambda pas: "Gadla Henry Mphakanyiswa Mandela" in pas, candidates))
        finally:
            shutil.rmtree('tmpindex')
