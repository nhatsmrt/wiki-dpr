from wiki_passage_retriever.retrieve import get_relevance_scores, get_most_relevant_passages, get_most_relevant_spans_from_wiki


class TestRelevanceScoring:
    def test_relevance_score(self):
        scores = get_relevance_scores(
            [
                "Nelson Rolihlahla Mandela was a South African anti-apartheid revolutionary, political leader and philanthropist who served as President of South Africa from 1994 to 1999",
                "Widely regarded as an icon of democracy and social justice, he received more than 250 honours, including the Nobel Peace Prize.",
                "Although critics on the right denounced him as a communist terrorist and those on the far left deemed him too eager to negotiate and reconcile with apartheid's supporters, he gained international acclaim for his activism."
            ],
            ["Nelson Mandela", "Nelson Mandela", "Nelson Mandela"],
            "Who is Nelson Mandela?"
        )

        return scores.argmax() == 0


class TestRetrieval:
    def test_retrieve_most_relevance(self):
        assert "was a South African anti-apartheid revolutionary" in get_most_relevant_passages("Nelson Mandela", "Who is Nelson Mandela?")[0]

    def test_topk(self):
        candidates = get_most_relevant_passages("Nelson Mandela", "Who is Nelson Mandela's father?", 5)
        assert len(candidates) == 5
        assert list(filter(lambda pas: "Gadla Henry Mphakanyiswa Mandela" in pas, candidates))  # check that the retriever finds the result


class TestSpan:
    def test_span(self):
        question = "Who was Nelson Mandela's father"
        spans = get_most_relevant_spans_from_wiki("Nelson Mandela", question, 5)
        assert len(spans) == 5
        assert spans[0] == "gadla henry mphakanyiswa mandela"
