"""Manual create_atom / replace_atom — dashboard write path."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from octop_memory.core import Memory
from octop_memory.storage.backends.sqlite import SqliteMemoryBackend
from octop_memory.storage.vector import VectorIndex
from octop_memory.types import AtomCard, Candidate, DigestRecord, Entity, Episode, JournalEntry

PG_DSN = os.environ.get("TEST_POSTGRES_DSN", "postgresql://localhost/octop_memory_test")


def _now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture(params=["sqlite", "postgres"])
def memory(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Memory]:
    """Run the same public write assertions against both production backends."""
    if request.param == "sqlite":
        mem = Memory(namespace="atom-write", backend_config={"db_path": str(tmp_path / "write.sqlite")})
        try:
            yield mem
        finally:
            mem.backend.close()
        return

    request.getfixturevalue("postgres_dsn")
    namespace = f"atom_write_{uuid.uuid4().hex[:12]}"
    mem = Memory(namespace=namespace, backend="postgres", backend_config={"dsn": PG_DSN})
    try:
        yield mem
    finally:
        backend = mem.backend
        if not backend._closed:  # type: ignore[attr-defined]
            backend.purge_namespace()  # type: ignore[attr-defined]
        backend.close()


def _seed_entity(memory: Memory, entity_id: str = "ent-1", name: str = "User") -> Entity:
    entity = Entity(
        id=entity_id,
        entity_type="User",
        canonical_name=name,
        aliases=[name],
        atom_count=0,
        created_at=_now(),
    )
    memory.add_entity(entity)
    return entity


def _seed_atom(
    memory: Memory,
    *,
    atom_id: str = "a1",
    entity_id: str = "ent-1",
    assertion: str = "likes coffee",
) -> AtomCard:
    cand = Candidate(
        id=f"c-{atom_id}",
        raw_event_ids=[],
        candidate_type="Preference",
        status="promoted",
        title=assertion,
        assertion=assertion,
        verbatim_quote=assertion,
        quote_event_id="",
        subject_name="User",
        subject_entity_type="User",
        target_entity_id=entity_id,
        confidence="high",
        importance="medium",
        recommended_action="promote",
        promotion_reason="seed",
        extractor_version="test",
        created_at=_now(),
        decided_at=_now(),
        decided_by="user",
    )
    memory.add_candidate(cand)
    atom = AtomCard(
        id=atom_id,
        entity_id=entity_id,
        candidate_id=cand.id,
        raw_event_ids=[],
        assertion=assertion,
        verbatim_quote=assertion,
        quote_event_id="",
        search_terms=["coffee"],
        occurred_at=_now(),
        confidence="high",
        importance="medium",
        created_at=_now(),
    )
    memory.add_atom(atom)
    memory.bump_entity_atom_count(entity_id, delta=1)
    return atom


class TestCreateAtom:
    def test_creates_entity_and_atom(self, memory: Memory) -> None:
        atom, entity, created = memory.create_atom(
            "喜欢喝美式咖啡",
            entity_name="饮品偏好",
            entity_type="Fact",
            kind="Preference",
        )
        assert created is True
        assert entity.canonical_name == "饮品偏好"
        assert entity.atom_count == 1
        assert atom.assertion == "喜欢喝美式咖啡"
        assert atom.entity_id == entity.id
        assert atom.deprecated_at is None

        cand = memory.get_candidate(atom.candidate_id)
        assert cand is not None
        assert cand.candidate_type == "Preference"
        assert cand.decided_by == "user"

        leaf = memory.backend.get_node_by_atom_id(atom.id)
        assert leaf is not None
        assert leaf.level == "leaf"
        parent = memory.get(leaf.parent_id) if leaf.parent_id else None
        assert parent is not None
        assert parent.level == "branch"
        assert parent.metadata.get("entity_id") == entity.id

        journal = memory.list_journal(action="create")
        assert len(journal) == 1
        assert journal[0].target_atom_id == atom.id

    def test_attaches_to_existing_entity(self, memory: Memory) -> None:
        existing = _seed_entity(memory)
        atom, entity, created = memory.create_atom(
            "住在上海",
            entity_id=existing.id,
            kind="Fact",
        )
        assert created is False
        assert entity.id == existing.id
        assert atom.entity_id == existing.id
        refreshed = memory.get_entity(existing.id)
        assert refreshed is not None
        assert refreshed.atom_count == 1

    def test_user_singleton_reuses_existing(self, memory: Memory) -> None:
        _seed_entity(memory, entity_id="ent-user", name="User")
        atom, entity, created = memory.create_atom(
            "真名叫小陈",
            entity_name="小陈",
            entity_type="User",
        )
        assert created is False
        assert entity.id == "ent-user"
        assert atom.entity_id == "ent-user"

    def test_rejects_empty_assertion(self, memory: Memory) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            memory.create_atom("   ", entity_name="X")

    def test_rejects_missing_entity(self, memory: Memory) -> None:
        with pytest.raises(ValueError, match="entity_id or entity_name"):
            memory.create_atom("a fact")
        with pytest.raises(ValueError, match="not found"):
            memory.create_atom("a fact", entity_id="missing")

    def test_rejects_duplicate_assertion(self, memory: Memory) -> None:
        memory.create_atom("喜欢喝美式咖啡", entity_name="饮品")
        with pytest.raises(ValueError, match="duplicate"):
            memory.create_atom("喜欢喝美式咖啡", entity_name="饮品", entity_type="Fact")


class TestReplaceAtom:
    def test_supersedes_and_relinks_tree(self, memory: Memory) -> None:
        _seed_entity(memory)
        old = _seed_atom(memory)
        raw_before = memory.list_raw(limit=100)
        new = memory.replace_atom(old.id, assertion="喜欢喝拿铁")
        assert new is not None
        assert new.id != old.id
        assert new.assertion == "喜欢喝拿铁"
        assert new.verbatim_quote == old.verbatim_quote
        assert new.entity_id == old.entity_id
        assert new.candidate_id == old.candidate_id
        assert new.raw_event_ids == old.raw_event_ids
        assert memory.list_raw(limit=100) == raw_before

        loaded_old = memory.get_atom(old.id)
        assert loaded_old is not None
        assert loaded_old.deprecated_at is not None
        assert loaded_old.superseded_by == new.id

        assert memory.backend.get_node_by_atom_id(old.id) is None
        new_leaf = memory.backend.get_node_by_atom_id(new.id)
        assert new_leaf is not None

        entity = memory.get_entity("ent-1")
        assert entity is not None
        assert entity.atom_count == 1

        live = memory.list_atoms(entity_id="ent-1", include_deprecated=False)
        assert [a.id for a in live] == [new.id]

        journal = memory.list_journal(action="user_edit")
        assert len(journal) == 1
        assert journal[0].before == {"assertion": "likes coffee", "atom_id": old.id}
        assert journal[0].after == {"assertion": "喜欢喝拿铁", "atom_id": new.id}
        assert journal[0].target_candidate_id is None

    def test_idempotent_when_text_unchanged(self, memory: Memory) -> None:
        _seed_entity(memory)
        old = _seed_atom(memory)
        same = memory.replace_atom(old.id, assertion="likes coffee")
        assert same is not None
        assert same.id == old.id
        assert memory.list_journal(action="user_edit") == []

    def test_unknown_or_deprecated_returns_none(self, memory: Memory) -> None:
        assert memory.replace_atom("nope", assertion="x") is None
        _seed_entity(memory)
        old = _seed_atom(memory)
        memory.deprecate_atom(old.id)
        assert memory.replace_atom(old.id, assertion="new text") is None

    def test_rejects_duplicate_of_other_live_atom(self, memory: Memory) -> None:
        _seed_entity(memory)
        _seed_atom(memory, atom_id="a1", assertion="likes coffee")
        other = _seed_atom(memory, atom_id="a2", assertion="likes tea")
        with pytest.raises(ValueError, match="duplicate"):
            memory.replace_atom(other.id, assertion="likes coffee")

    def test_concurrent_replacement_rolls_back_successor(self, memory: Memory, monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_entity(memory)
        old = _seed_atom(memory)
        monkeypatch.setattr(memory, "supersede_atom", lambda *args, **kwargs: False)

        with pytest.raises(ValueError, match="replaced concurrently"):
            memory.replace_atom(old.id, assertion="new text")

        atoms = memory.list_atoms(entity_id=old.entity_id, include_deprecated=True)
        assert [atom.id for atom in atoms] == [old.id]
        assert memory.list_journal(action="user_edit") == []

    def test_active_atom_can_only_choose_one_successor(self, memory: Memory) -> None:
        _seed_entity(memory)
        old = _seed_atom(memory)
        first = _seed_atom(memory, atom_id="a2", assertion="first successor")
        second = _seed_atom(memory, atom_id="a3", assertion="second successor")

        assert memory.supersede_atom(old.id, new_atom_id=first.id) is True
        assert memory.supersede_atom(old.id, new_atom_id=second.id) is False
        loaded = memory.get_atom(old.id)
        assert loaded is not None
        assert loaded.superseded_by == first.id


class TestForgetAtom:
    def test_fact_content_does_not_enter_application_logs(
        self, memory: Memory, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="octop_memory"):
            atom, _, _ = memory.create_atom("A calendar is private", entity_name="Private label")
            leaf = memory.store("A schedule is personal", topic="Personal label")
        assert atom.id in caplog.text and leaf.atom_id is not None and leaf.atom_id in caplog.text
        assert "A calendar is private" not in caplog.text
        assert "A schedule is personal" not in caplog.text
        assert "Private label" not in caplog.text
        assert "Personal label" not in caplog.text

    def test_limits_cleanup_to_the_same_namespace(self, memory: Memory, tmp_path: Path) -> None:
        target, entity, _ = memory.create_atom("Read a printed calendar", entity_name="Calendar")
        source = memory.get_raw(target.quote_event_id)
        candidate = memory.get_candidate(target.candidate_id)
        assert source is not None and candidate is not None
        namespace = f"other_{uuid.uuid4().hex}"
        if isinstance(memory.backend, SqliteMemoryBackend):
            other = Memory(namespace=namespace, backend_config={"db_path": tmp_path / "write.sqlite"})
        else:
            other = Memory(namespace=namespace, backend="postgres", backend_config={"dsn": PG_DSN})
        try:
            other.add_raw_batch([source])
            other.add_candidate(candidate)
            other.add_entity(entity)
            other.add_atom(target)
            assert memory.forget_atom(target.id) is True
            assert memory.get_atom(target.id) is None
            assert other.get_atom(target.id) == target
            assert other.get_candidate(candidate.id) == candidate
            assert other.get_raw(source.id) == source
        finally:
            if not isinstance(other.backend, SqliteMemoryBackend):
                other.backend.purge_namespace()  # type: ignore[attr-defined]
            other.backend.close()

    def test_rolls_back_relational_cleanup_if_vector_deletion_fails(
        self, memory: Memory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        atom, entity, _ = memory.create_atom("Use a ruler to draw straight lines", entity_name="Drawing")
        vector_index = Mock(spec=VectorIndex)
        vector_index.delete.side_effect = RuntimeError("vector unavailable")
        monkeypatch.setattr(memory, "_vector_index", vector_index)
        with pytest.raises(RuntimeError, match="vector unavailable"):
            memory.forget_atom(atom.id)
        assert memory.get_atom(atom.id) == atom
        assert memory.get_raw(atom.quote_event_id) is not None
        assert memory.get_candidate(atom.candidate_id) is not None
        assert memory.get_entity(entity.id) == entity

    def test_forgets_fact_history_and_summary_without_removing_other_fact(self, memory: Memory) -> None:
        old, entity, _ = memory.create_atom("Favorite drink is jasmine tea", entity_name="Drinks")
        revised = memory.replace_atom(old.id, assertion="Favorite drink is peppermint tea")
        assert revised is not None
        other, _, _ = memory.create_atom("Drink water before practice", entity_id=entity.id)
        page = memory.get_entity_page(entity.id)
        assert page is not None
        assert memory.apply_entity_page_regen(
            entity.id,
            summary_markdown="Peppermint tea is the favorite.",
            headline="Peppermint tea",
            topics=["peppermint"],
            when=_now(),
            expected_mutation_version=page.mutation_version,
        )

        assert memory.atom_forget_ids(revised.id) == (revised.id, old.id)
        assert memory.forget_atom(revised.id) is True
        assert memory.forget_atom(revised.id) is False
        assert memory.atom_forget_ids(revised.id) == ()
        assert memory.get_atom(old.id) is None
        assert memory.get_atom(revised.id) is None
        assert memory.search_atoms("peppermint", include_deprecated=True) == []
        assert memory.search_atoms("jasmine", include_deprecated=True) == []
        assert memory.recall("peppermint") == []
        assert memory.get_raw(old.quote_event_id) is None
        assert memory.search_raw("jasmine") == []
        assert memory.get_candidate(old.candidate_id) is None
        assert memory.list_journal(target_atom_id=revised.id) == []
        assert memory.list_journal(target_atom_id=old.id) == []
        assert memory.get_atom(other.id) == other
        remaining = memory.get_entity(entity.id)
        assert remaining is not None and remaining.atom_count == 1
        clean = memory.get_entity_page(entity.id)
        assert clean is not None and clean.dirty and not clean.summary_markdown and not clean.headline

    def test_cleans_merged_sources_and_shared_derivatives(self, memory: Memory) -> None:
        atom, entity, _ = memory.create_atom("Use a paper planner", entity_name="Planner")
        candidate = memory.get_candidate(atom.candidate_id)
        assert candidate is not None
        branch = memory.get_or_create_entity_branch(entity.id)
        other_leaf = memory.store("Buy a second ruler", topic="Stationery", parent_id=branch.id)
        merged = memory.add_raw("The paper planner is important", event_type="user_message")
        other_source = memory.add_raw("Buy new stationery", event_type="user_message")
        memory.add_candidate(
            replace(
                candidate,
                id="merged-candidate",
                raw_event_ids=[merged.id],
                quote_event_id=merged.id,
            )
        )
        memory.append_journal(
            JournalEntry(
                id="merge-log",
                timestamp=_now(),
                action="merge",
                actor="rule",
                target_entity_id=entity.id,
                target_atom_id=atom.id,
                target_candidate_id="merged-candidate",
            )
        )
        related = Episode(
            id="planner-episode",
            raw_event_ids=[atom.quote_event_id],
            occurred_at=_now(),
            summary="User brought a paper planner.",
            verbatim_quote=atom.verbatim_quote,
            quote_event_id=atom.quote_event_id,
            emotion="neutral",
            intensity=1,
            people=[],
            topics=[],
            extractor_version="test",
            created_at=_now(),
            digest_ids=["day-digest"],
        )
        other = replace(
            related,
            id="stationery-episode",
            raw_event_ids=[other_source.id],
            quote_event_id=other_source.id,
            summary=other_source.content,
            verbatim_quote=other_source.content,
        )
        memory.add_episodes([related, other])
        day = DigestRecord(
            id="day-digest",
            period_kind="daily",
            period_key="2026-06-01",
            period_start=datetime(2026, 6, 1, tzinfo=UTC),
            period_end=datetime(2026, 6, 2, tzinfo=UTC),
            markdown=f"{related.summary} {other.summary}",
            episode_ids=[related.id, other.id],
            llm_version="test",
            created_at=_now(),
            updated_at=_now(),
        )
        memory.upsert_digest(day)
        memory.upsert_digest(
            replace(
                day,
                id="rollup-digest",
                period_kind="weekly",
                period_key="2026-W23",
                period_end=datetime(2026, 6, 8, tzinfo=UTC),
                episode_ids=[],
            )
        )

        assert memory.forget_atom(atom.id) is True
        assert memory.get_entity(entity.id) is None
        assert memory.find_entity_by_alias("planner") is None
        assert memory.get_entity_page(entity.id) is None
        assert not any(node.metadata.get("entity_id") == entity.id for node in memory.get_tree())
        assert memory.get(other_leaf.id) == replace(other_leaf, parent_id=branch.parent_id)
        assert memory.get_raw(merged.id) is None
        assert memory.get_candidate("merged-candidate") is None
        assert memory.get_episode(related.id) is None
        assert memory.search_episodes("planner") == []
        assert memory.get_digest("daily", day.period_key) is None
        assert memory.get_digest("weekly", "2026-W23") is None
        assert memory.get_episode(other.id) == replace(other, digest_ids=[])
        assert memory.get_raw(other_source.id) == other_source

    def test_forgetting_old_fact_detaches_its_provenance_from_surviving_revision(self, memory: Memory) -> None:
        old, entity, _ = memory.create_atom("Prefer bright red folders", entity_name="Folders")
        revised = memory.replace_atom(old.id, assertion="Prefer blue folders")
        assert revised is not None
        assert revised.verbatim_quote == old.assertion

        assert memory.forget_atom(old.id) is True
        assert memory.get_atom(old.id) is None
        assert memory.get_raw(old.quote_event_id) is None
        assert memory.get_candidate(old.candidate_id) is None
        assert memory.list_journal(target_atom_id=revised.id) == []
        kept = memory.get_atom(revised.id)
        assert kept is not None and kept.assertion == "Prefer blue folders"
        assert not kept.candidate_id
        assert not kept.raw_event_ids
        assert not kept.quote_event_id and not kept.verbatim_quote
        remaining = memory.get_entity(entity.id)
        assert remaining is not None and remaining.atom_count == 1

    def test_includes_predecessor_no_longer_present_in_live_atoms(self, memory: Memory) -> None:
        current, entity, _ = memory.create_atom("Use a blue label", entity_name="Labels")
        memory.append_journal(
            JournalEntry(
                id="historical-edit",
                timestamp=_now(),
                action="user_edit",
                actor="user",
                target_entity_id=entity.id,
                target_atom_id=current.id,
                before={"atom_id": "archived-predecessor", "assertion": "Use a red label"},
                after={"atom_id": current.id, "assertion": current.assertion},
            )
        )
        assert memory.get_atom("archived-predecessor") is None
        assert memory.atom_forget_ids(current.id) == (current.id, "archived-predecessor")
        assert memory.forget_atom(current.id)
        assert memory.list_journal(target_atom_id=current.id) == []

    def test_surviving_revision_drops_terms_inherited_from_forgotten_source(self, memory: Memory) -> None:
        entity = _seed_entity(memory, name="Breaks")
        old = _seed_atom(memory, entity_id=entity.id, assertion="Take a coffee break")
        revised = memory.replace_atom(old.id, assertion="Take a tea break")
        assert revised is not None
        assert revised.search_terms == ["coffee"]
        assert memory.forget_atom(old.id)
        kept = memory.get_atom(revised.id)
        assert kept is not None and kept.assertion == "Take a tea break"
        assert kept.search_terms == []
        assert memory.search_atoms("coffee") == []
