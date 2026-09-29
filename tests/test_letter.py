from dataclasses import replace

import pytest

from agent_emploi.agents.letter import (
    MAX_DESCRIPTION_CHARS,
    LetterAgent,
    build_prompt,
    clean,
)
from agent_emploi.config import LetterConfig
from agent_emploi.llm.router import Router
from agent_emploi.models import FitVerdict, Job, Letter
from agent_emploi.profile import CvText
from tests.conftest import ScriptedProvider

FIT = FitVerdict(
    score=78,
    verdict="apply",
    matched=["Python", "LLM"],
    gaps=["Kubernetes"],
    reason="Bon recouvrement technique.",
    language="fr",
)

LETTRE = " ".join(["mot"] * 170)


def make_job(**fields) -> Job:
    base = {
        "title": "Ingénieur LLM",
        "company": "Acme",
        "location": "Paris",
        "contract": "stage",
        "description": "Développement d'agents LLM en Python.",
    }
    return Job.build(source="fake", url="https://example.com/jobs/1", **{**base, **fields})


def make_agent(config, profile, responses: list[str], monkeypatch) -> LetterAgent:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    router = Router(config)
    router._providers["anthropic"] = ScriptedProvider(responses)
    return LetterAgent(router, profile, config.letter)


@pytest.fixture
def agent(config, profile, monkeypatch) -> LetterAgent:
    return make_agent(config, profile, [LETTRE], monkeypatch)


class TestSystemPrompt:
    def test_carries_cv_voice_and_banned_list(self, agent):
        assert "Perceptron multicouche" in agent.system
        assert "Je vouvoie toujours" in agent.system
        assert "fort de mon expérience" in agent.system

    def test_falls_back_when_voice_is_missing(self, config, profile):
        mute = LetterAgent(None, replace(profile, voice=""), config.letter)
        assert "Aucun échantillon d'écriture" in mute.system

    def test_states_the_configured_length(self, config, profile):
        agent = LetterAgent(None, profile, LetterConfig(min_words=120, max_words=140))
        assert "entre 120 et 140 mots" in agent.system

    def test_holds_no_offer_so_the_cache_prefix_stays_stable(self, agent):
        assert "Acme" not in agent.system


class TestBuildPrompt:
    def test_carries_offer_and_fit_analysis(self):
        prompt = build_prompt(make_job(), FIT)
        assert "Ingénieur LLM" in prompt
        assert "Acme" in prompt
        assert "Python, LLM" in prompt
        assert "Kubernetes" in prompt

    def test_names_the_language_of_the_offer(self):
        assert "en anglais" in build_prompt(make_job(), FIT.model_copy(update={"language": "en"}))
        assert "en français" in build_prompt(make_job(), FIT)

    def test_truncates_a_long_description(self):
        prompt = build_prompt(make_job(description="mot " * 6000), FIT)
        assert "description tronquée" in prompt
        assert len(prompt) < MAX_DESCRIPTION_CHARS + 2000

    def test_appends_feedback_on_a_retry(self):
        prompt = build_prompt(make_job(), FIT, "trop longue : 300 mots")
        assert "# Reprise" in prompt
        assert "trop longue" in prompt


class TestClean:
    def test_strips_code_fences(self):
        assert clean("```\nMadame, Monsieur,\n```") == "Madame, Monsieur,"

    def test_strips_a_subject_line(self):
        assert clean("Objet : candidature\n\nMadame,") == "Madame,"
        assert clean("Subject: application\n\nDear,") == "Dear,"

    def test_strips_surrounding_quotes(self):
        assert clean('"Madame, Monsieur,"') == "Madame, Monsieur,"

    def test_leaves_a_clean_letter_untouched(self):
        assert clean("Madame,\n\nJ'ai écrit un MLP.") == "Madame,\n\nJ'ai écrit un MLP."


class TestWrite:
    def test_returns_the_letter_in_the_offer_language(self, agent):
        letter = agent.write(make_job(), FIT)
        assert letter.text == LETTRE
        assert letter.language == "fr"
        assert letter.word_count == 170
        assert not letter.regenerated

    def test_uses_the_paid_route_with_a_cacheable_system_block(self, agent):
        agent.write(make_job(), FIT)
        call = agent.router._providers["anthropic"].calls[0]
        assert call["model"] == "claude-opus-5"
        assert call["system"] == agent.system

    def test_reports_banned_phrases_without_refusing_the_letter(
        self, config, profile, monkeypatch
    ):
        agent = make_agent(
            config, profile, ["Fort de mon experience, je vous écris."], monkeypatch
        )
        letter = agent.write(make_job(), FIT)
        assert letter.banned_hits == ["fort de mon expérience"]

    def test_marks_a_letter_written_from_feedback(self, agent):
        assert agent.write(make_job(), FIT, feedback="reprends").regenerated

    def test_logs_usage_against_the_job(self, agent, config):
        job = make_job()
        agent.write(job, FIT)
        line = config.paths.usage_file.read_text(encoding="utf-8").splitlines()[-1]
        assert f'"job_id":"{job.id}"' in line.replace(" ", "")
        assert '"task":"letter"' in line.replace(" ", "")


class TestDefects:
    def letter(self, text: str, **fields) -> Letter:
        return Letter(text=text, language="fr", **fields)

    def test_none_on_a_correct_letter(self, agent):
        assert agent.defects(self.letter(LETTRE)) == []

    def test_flags_a_short_letter(self, agent):
        [defect] = agent.defects(self.letter("mot " * 40))
        assert "trop courte" in defect

    def test_flags_a_long_letter(self, agent):
        [defect] = agent.defects(self.letter("mot " * 400))
        assert "trop longue" in defect

    def test_flags_banned_phrases(self, agent):
        defects = agent.defects(
            self.letter(LETTRE, banned_hits=["passionné par"])
        )
        assert "passionné par" in defects[0]


class TestVariants:
    def test_letter_is_written_from_the_chosen_cv(self, config, profile, monkeypatch):
        profile = replace(
            profile,
            cvs=(CvText("dev", "Dev", "Backend Django."), CvText("ml", "ML", "Keras, vision.")),
        )
        agent = make_agent(config, profile, [LETTRE], monkeypatch)
        letter = agent.write(make_job(), FIT.model_copy(update={"cv": "ml"}))
        system = agent.router._providers["anthropic"].calls[0]["system"]
        assert "Keras, vision." in system
        assert "Backend Django." not in system
        assert letter.cv == "ml"
