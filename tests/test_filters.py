from datetime import datetime, timedelta, timezone

import pytest

from agent_emploi.config import FiltersConfig, SearchConfig
from agent_emploi.filters import (
    LexicalScorer,
    contains_any,
    fold,
    screen_content,
    screen_metadata,
    tokenize,
)
from agent_emploi.models import Job

CV = """
Étudiant à l'école 42, je cherche un stage en machine learning.
Compétences : Python, Tensorflow, Pandas, numpy, Docker, PostgreSQL, C, C++.
IA générative : LLM, RAG, agents. Deep learning, réseaux de neurones,
apprentissage par renforcement, computer vision.
"""


def make_job(**fields) -> Job:
    base = {
        "title": "Ingénieur machine learning",
        "company": "Acme",
        "description": "",
    }
    return Job.build(source="fake", url="https://example.com/jobs/1", **{**base, **fields})


class TestTokenize:
    def test_folds_accents_and_case(self):
        assert fold("Ingénieur IA") == "ingenieur ia"

    def test_drops_stopwords_and_numbers(self):
        assert tokenize("le poste de data scientist en 2026") == ["data", "scientist"]

    def test_keeps_technology_names(self):
        assert "c++" in tokenize("Développement C++ et node.js")
        assert "node.js" in tokenize("Développement C++ et node.js")


class TestLexicalScorer:
    def test_close_offer_scores_higher_than_unrelated(self):
        scorer = LexicalScorer(CV)
        proche = scorer.score(
            "Stage machine learning : entraînement de modèles deep learning en "
            "Python avec Tensorflow, mise en production Docker."
        )
        loin = scorer.score(
            "Nous recherchons un commercial terrain pour la vente de mobilier "
            "de bureau auprès des collectivités."
        )
        assert proche > loin
        assert 0.0 <= loin < proche <= 1.0

    def test_empty_cv_scores_zero(self):
        assert LexicalScorer("").score("machine learning") == 0.0

    def test_empty_text_scores_zero(self):
        assert LexicalScorer(CV).score("") == 0.0

    def test_is_symmetric_and_deterministic(self):
        scorer = LexicalScorer(CV)
        text = "Python, Tensorflow, deep learning"
        assert scorer.score(text) == pytest.approx(scorer.score(text))
        assert LexicalScorer(text).score(CV) == pytest.approx(scorer.score(text))


class TestContainsAny:
    def test_matches_ignoring_accents_and_case(self):
        assert contains_any("Ingénieur IA", ["intelligence artificielle", "ia"]) == "ia"

    def test_prefix_term_matches_variants(self):
        assert contains_any("Data Scientist senior", ["data scien"]) == "data scien"

    def test_returns_none_without_match(self):
        assert contains_any("Développeur backend", ["llm", "nlp"]) is None


@pytest.fixture
def filters() -> FiltersConfig:
    return FiltersConfig(
        required_any=["machine learning", "ia"],
        excluded_any=["commercial"],
        min_lexical_score=0.05,
    )


@pytest.fixture
def search() -> SearchConfig:
    return SearchConfig(queries=["ia"], contracts=["CDI", "stage"], max_age_days=30)


class TestScreenMetadata:
    def test_accepts_a_relevant_offer(self, filters, search):
        assert screen_metadata(make_job(contract="stage"), filters, search).passed

    def test_rejects_excluded_term_in_title(self, filters, search):
        result = screen_metadata(make_job(title="Commercial IA"), filters, search)
        assert not result.passed
        assert result.reason == "exclu:commercial"

    def test_rejects_unwanted_contract(self, filters, search):
        result = screen_metadata(make_job(contract="freelance"), filters, search)
        assert result.reason == "contrat:freelance"

    def test_missing_contract_passes(self, filters, search):
        """Une métadonnée absente ne doit pas coûter une offre pertinente."""
        assert screen_metadata(make_job(contract=None), filters, search).passed

    def test_rejects_stale_offer(self, filters, search):
        now = datetime(2026, 8, 9, tzinfo=timezone.utc)
        job = make_job(posted_at=now - timedelta(days=45))
        result = screen_metadata(job, filters, search, now=now)
        assert result.reason.startswith("anciennete:45j")

    def test_rejects_off_topic_title_when_required_in_title(self, filters, search):
        result = screen_metadata(make_job(title="Développeur backend"), filters, search)
        assert result.reason == "titre_hors_sujet"

    def test_off_topic_title_passes_when_requirement_relaxed(self, filters, search):
        filters.require_in_title = False
        assert screen_metadata(make_job(title="Développeur backend"), filters, search).passed


class TestScreenContent:
    def test_accepts_offer_close_to_cv(self, filters):
        job = make_job(
            description="Stage machine learning : Python, Tensorflow, deep learning."
        )
        result = screen_content(job, filters, LexicalScorer(CV))
        assert result.passed
        assert result.lexical_score is not None and result.lexical_score > 0.05

    def test_rejects_missing_description(self, filters):
        result = screen_content(make_job(), filters, LexicalScorer(CV))
        assert result.reason == "description_absente"

    def test_rejects_when_no_required_term(self, filters):
        job = make_job(title="Poste", description="Vente de mobilier de bureau.")
        assert screen_content(job, filters, LexicalScorer(CV)).reason == "aucun_terme_requis"

    def test_rejects_below_lexical_threshold(self, filters):
        filters.min_lexical_score = 0.9
        job = make_job(description="Machine learning appliqué à la finance.")
        result = screen_content(job, filters, LexicalScorer(CV))
        assert not result.passed
        assert result.reason.startswith("lexical:")
        assert result.lexical_score is not None
