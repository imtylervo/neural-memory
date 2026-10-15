"""Regression: a cancelled caller must not kill the SQLite write worker."""

from __future__ import annotations

import asyncio

import pytest

from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def dialect(tmp_path):
    d = SQLiteDialect(tmp_path / "wq.db")
    await d.initialize()
    await d.execute_write("CREATE TABLE t (x INTEGER)")
    yield d
    await d.close()


async def test_cancelled_writer_does_not_kill_worker(dialect):
    task = asyncio.create_task(dialect.execute_write("INSERT INTO t VALUES (1)"))
    await asyncio.sleep(0)  # let the task enqueue and start awaiting its future
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The worker must still serve later writes.
    await asyncio.wait_for(dialect.execute_write("INSERT INTO t VALUES (2)"), timeout=5)
    assert dialect._write_task is not None and not dialect._write_task.done()
