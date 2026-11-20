"""Shared provenance cleanup for SQLite and PostgreSQL."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any


class AtomForgetStore:
    """Operate on one canonical namespace using the caller's transaction."""

    def __init__(self, execute: Callable[..., Any], namespace: str, *, postgres_schema: str = "") -> None:
        self._execute = execute
        self._namespace = namespace
        self._postgres_schema = postgres_schema

    def _run(
        self,
        statement: str,
        table: str,
        condition: str,
        *params: Any,
        set_sql: str = "",
        set_params: tuple[Any, ...] = (),
        suffix: str = "",
    ) -> Any:
        if self._postgres_schema:
            name = f"{self._postgres_schema}.{table}"
            condition = f"namespace = ? AND ({condition})"
            params = (self._namespace, *params)
        else:
            name = f"{self._namespace}_{table}"
        sql = f"{statement} {name}{set_sql} WHERE {condition}{suffix}"
        return self._execute(sql.replace("?", "%s") if self._postgres_schema else sql, (*set_params, *params))

    def _select(self, table: str, columns: str, where: str, *params: Any, suffix: str = "") -> list[Any]:
        return list(self._run(f"SELECT {columns} FROM", table, where, *params, suffix=suffix).fetchall())

    def _delete(self, table: str, where: str, *params: Any) -> None:
        self._run("DELETE FROM", table, where, *params)

    def _update(self, table: str, set_sql: str, values: tuple[Any, ...], where: str, *params: Any) -> None:
        self._run("UPDATE", table, where, *params, set_sql=f" SET {set_sql}", set_params=values)

    def _in(self, column: str) -> str:
        if self._postgres_schema:
            return f"({column}) = ANY(?)"
        return f"({column}) IN (SELECT value FROM json_each(?))"

    def _overlaps(self, column: str) -> str:
        if self._postgres_schema:
            source = f"jsonb_array_elements_text({column}) AS ids(value)"
        else:
            source = f"json_each({column})"
        return f"EXISTS (SELECT 1 FROM {source} WHERE {self._in('value')})"

    def _ids_param(self, ids: Iterable[str]) -> list[str] | str:
        values = list(ids)
        return values if self._postgres_schema else json.dumps(values)

    def _by_source(self) -> str:
        return f"{self._in('quote_event_id')} OR {self._overlaps('raw_event_ids')}"

    @staticmethod
    def _json_ids(value: str | list[str]) -> list[str]:
        return list(json.loads(value)) if isinstance(value, str) else value

    def history(self, atom_id: str, *, lock: bool = False) -> tuple[tuple[str, ...], list[Any]]:
        """Return IDs from atom links and retained journal, plus surviving rows."""
        columns = "id, entity_id, candidate_id, raw_event_ids, quote_event_id"
        row_lock = " FOR UPDATE" if lock and self._postgres_schema else ""
        rows = self._select("atoms", columns, "id = ?", atom_id, suffix=row_lock)
        if not rows:
            return (), []
        ids = [atom_id]
        pending = [atom_id]
        visited = {atom_id}
        logged_successor = self._in("after ->> 'superseded_by'")
        while pending:
            successors = self._ids_param(pending)
            revisions = self._select(
                "journal",
                "CASE WHEN action = 'user_edit' THEN before ->> 'atom_id' ELSE target_atom_id END AS predecessor",
                f"(action = 'user_edit' AND {self._in('target_atom_id')}) OR "
                f"(action = 'consolidate' AND {logged_successor})",
                successors,
                successors,
            )
            journal_ids: set[str] = {row["predecessor"] for row in revisions if row["predecessor"]} - visited
            predecessors = self._select(
                "atoms",
                columns,
                f"{self._in('superseded_by')} OR {self._in('id')}",
                successors,
                self._ids_param(journal_ids),
                suffix=f" ORDER BY id{row_lock}",
            )
            rows.extend(row for row in predecessors if row["id"] not in visited)
            pending = sorted((journal_ids | {row["id"] for row in predecessors}) - visited)
            visited.update(pending)
            ids.extend(pending)
        return tuple(ids), rows

    def forget(self, atom_id: str) -> tuple[str, ...]:
        atom_ids, history = self.history(atom_id, lock=True)
        if not atom_ids:
            return ()

        atoms = self._ids_param(atom_ids)
        sources: set[str] = set()
        for row in history:
            sources.update(self._json_ids(row["raw_event_ids"]))
            sources.add(row["quote_event_id"])
        merged = self._select(
            "journal",
            "target_candidate_id",
            f"action = 'merge' AND target_candidate_id IS NOT NULL AND {self._in('target_atom_id')}",
            atoms,
        )
        direct_candidates = {row["candidate_id"] for row in history} | {row["target_candidate_id"] for row in merged}
        direct_candidates.discard("")
        demoted_atom = self._in("payload ->> 'demoted_atom_id'")
        candidates = self._select(
            "candidates",
            "id, raw_event_ids, quote_event_id",
            f"{self._in('id')} OR {demoted_atom}",
            self._ids_param(direct_candidates),
            atoms,
        )
        for row in candidates:
            sources.update(self._json_ids(row["raw_event_ids"]))
            sources.add(row["quote_event_id"])
        events = self._ids_param(sources - {""})
        candidate_ids = direct_candidates | {row["id"] for row in candidates}
        related_candidates = self._select("candidates", "id", self._by_source(), events, events)
        candidate_ids.update(row["id"] for row in related_candidates)
        candidates_param = self._ids_param(candidate_ids)

        self._delete("memory_nodes", self._in("atom_id"), atoms)
        self._delete("atoms", self._in("id"), atoms)
        self._delete("raw_events", self._in("id"), events)
        self._delete("candidates", self._in("id"), candidates_param)
        journal_atom_columns = (
            "target_atom_id",
            "before ->> 'atom_id'",
            "after ->> 'atom_id'",
            "after ->> 'superseded_by'",
        )
        self._delete(
            "journal",
            " OR ".join(self._in(column) for column in (*journal_atom_columns, "target_candidate_id")),
            *((atoms,) * len(journal_atom_columns)),
            candidates_param,
        )

        affected_entities = {row["entity_id"] for row in history}
        affected_entities.update(self._unlink_shared_sources(sources, candidate_ids))
        self._delete_related_episodes(events)
        for entity_id in affected_entities:
            self._refresh_entity(entity_id)
        return atom_ids

    def _unlink_shared_sources(self, erased_events: set[str], erased_candidates: set[str]) -> set[str]:
        """Preserve unrelated assertions even if their supporting source was shared."""
        events = self._ids_param(erased_events - {""})
        candidates = self._ids_param(erased_candidates)
        shared = self._select(
            "atoms",
            "id, entity_id, raw_event_ids, quote_event_id, candidate_id",
            f"{self._in('candidate_id')} OR {self._by_source()}",
            candidates,
            events,
            events,
        )
        json_value = "CAST(? AS jsonb)" if self._postgres_schema else "?"
        for row in shared:
            candidate = row["candidate_id"]
            quote = row["quote_event_id"]
            source_ids = self._json_ids(row["raw_event_ids"])
            candidate_erased = candidate in erased_candidates
            clear_derived = (
                candidate_erased or quote in erased_events or (not quote and not erased_events.isdisjoint(source_ids))
            )
            self._update(
                "atoms",
                f"candidate_id = ?, raw_event_ids = {json_value}, quote_event_id = ?, "
                "verbatim_quote = CASE WHEN ? THEN '' ELSE verbatim_quote END, "
                "search_terms = CASE WHEN ? THEN '[]' ELSE search_terms END",
                (
                    "" if candidate_erased else candidate,
                    json.dumps([event_id for event_id in source_ids if event_id not in erased_events]),
                    "" if clear_derived else quote,
                    clear_derived,
                    clear_derived,
                ),
                "id = ?",
                row["id"],
            )
        return {row["entity_id"] for row in shared}

    def _delete_related_episodes(self, events: list[str] | str) -> None:
        episodes = self._select("episodes", "id, digest_ids", self._by_source(), events, events)
        if not episodes:
            return
        episode_ids = self._ids_param(row["id"] for row in episodes)
        direct_digests = self._ids_param(digest for row in episodes for digest in self._json_ids(row["digest_ids"]))
        affected_digests = self._select(
            "digests",
            "id, period_kind, period_start",
            f"{self._in('id')} OR {self._overlaps('episode_ids')}",
            direct_digests,
            episode_ids,
        )
        self._delete("episodes", self._in("id"), episode_ids)
        if not affected_digests:
            return
        digest_ids = {row["id"] for row in affected_digests}
        # Weekly/monthly digests can summarize stored daily digests in
        # addition to the episodes recorded on the roll-up itself.
        for day in (row["period_start"] for row in affected_digests if row["period_kind"] == "daily"):
            rollups = self._select(
                "digests",
                "id",
                "period_kind IN ('weekly', 'monthly') AND period_start <= ? AND period_end > ?",
                day,
                day,
            )
            digest_ids.update(row["id"] for row in rollups)
        digests = self._ids_param(digest_ids)
        self._delete("digests", self._in("id"), digests)
        remaining = self._select("episodes", "id, digest_ids", self._overlaps("digest_ids"), digests)
        json_value = "CAST(? AS jsonb)" if self._postgres_schema else "?"
        for row in remaining:
            self._update(
                "episodes",
                f"digest_ids = {json_value}",
                (json.dumps([digest for digest in self._json_ids(row["digest_ids"]) if digest not in digest_ids]),),
                "id = ?",
                row["id"],
            )

    def _refresh_entity(self, entity_id: str) -> None:
        counts = self._select(
            "atoms",
            "COUNT(*) AS total, COUNT(*) FILTER (WHERE deprecated_at IS NULL AND superseded_by IS NULL) AS live",
            "entity_id = ?",
            entity_id,
        )[0]
        if not counts["total"] and not self._select(
            "candidates", "id", "target_entity_id = ?", entity_id, suffix=" LIMIT 1"
        ):
            for table, field in (
                ("aliases", "entity_id"),
                ("entity_pages", "entity_id"),
                ("thread_active_entities", "entity_id"),
                ("journal", "target_entity_id"),
                ("entities", "id"),
            ):
                self._delete(table, f"{field} = ?", entity_id)
            branches = self._select(
                "memory_nodes", "id, parent_id", "level = 'branch' AND metadata ->> 'entity_id' = ?", entity_id
            )
            for branch in branches:
                self._update("memory_nodes", "parent_id = ?", (branch["parent_id"],), "parent_id = ?", branch["id"])
                self._delete("memory_nodes", "id = ?", branch["id"])
            return

        self._update("entities", "atom_count = ?", (counts["live"],), "id = ?", entity_id)
        now = datetime.now(UTC)
        self._update(
            "entity_pages",
            "summary_markdown = '', headline = '', topics = '[]', dirty = TRUE, regen_attempt_count = 0, "
            "mutation_version = mutation_version + 1, last_regen_at = NULL, last_user_edit_at = NULL, updated_at = ?",
            (now if self._postgres_schema else now.isoformat(),),
            "entity_id = ?",
            entity_id,
        )
