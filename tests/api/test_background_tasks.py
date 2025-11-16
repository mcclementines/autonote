"""Unit tests for background task queue."""

import asyncio

import pytest

from api.services.background_tasks import BackgroundTaskQueue


@pytest.mark.asyncio
class TestBackgroundTaskQueue:
    """Test suite for BackgroundTaskQueue."""

    async def test_queue_initialization(self):
        """Test queue initializes correctly."""
        queue = BackgroundTaskQueue()

        assert queue.queue is not None
        assert queue.worker_task is None
        assert queue._running is False

    async def test_start_worker(self):
        """Test starting the background worker."""
        queue = BackgroundTaskQueue()

        await queue.start_worker()

        assert queue._running is True
        assert queue.worker_task is not None

        # Cleanup
        await queue.stop_worker()

    async def test_stop_worker(self):
        """Test stopping the background worker."""
        queue = BackgroundTaskQueue()

        await queue.start_worker()
        await queue.stop_worker()

        assert queue._running is False

    async def test_start_worker_idempotent(self):
        """Test that starting worker multiple times doesn't create multiple workers."""
        queue = BackgroundTaskQueue()

        await queue.start_worker()
        first_task = queue.worker_task

        await queue.start_worker()  # Should not create new worker
        second_task = queue.worker_task

        assert first_task == second_task

        # Cleanup
        await queue.stop_worker()

    async def test_stop_worker_when_not_running(self):
        """Test stopping worker when it's not running."""
        queue = BackgroundTaskQueue()

        # Should not raise error
        await queue.stop_worker()

        assert queue._running is False

    async def test_enqueue_task(self):
        """Test enqueueing a task."""
        queue = BackgroundTaskQueue()

        # Simple async function
        async def test_func(value):
            return value * 2

        queue.enqueue(test_func, 5)

        # Queue should have 1 item
        assert queue.queue.qsize() == 1

    async def test_task_execution(self):
        """Test that enqueued tasks are executed."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        # Shared state to verify execution
        executed = []

        async def test_task(value):
            executed.append(value)

        # Enqueue tasks
        queue.enqueue(test_task, 1)
        queue.enqueue(test_task, 2)
        queue.enqueue(test_task, 3)

        # Wait for tasks to complete
        await queue.wait_for_completion()

        assert executed == [1, 2, 3]

        # Cleanup
        await queue.stop_worker()

    async def test_task_with_kwargs(self):
        """Test executing task with keyword arguments."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        result = []

        async def test_task(a, b=None):
            result.append((a, b))

        queue.enqueue(test_task, 10, b=20)

        await queue.wait_for_completion()

        assert result == [(10, 20)]

        # Cleanup
        await queue.stop_worker()

    async def test_task_error_handling(self):
        """Test that errors in tasks don't crash the worker."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        executed = []

        async def failing_task():
            raise ValueError("Test error")

        async def success_task():
            executed.append("success")

        # Enqueue failing task followed by success task
        queue.enqueue(failing_task)
        queue.enqueue(success_task)

        # Wait for tasks
        await queue.wait_for_completion()

        # Worker should still be running and success task should execute
        assert queue._running is True
        assert "success" in executed

        # Cleanup
        await queue.stop_worker()

    async def test_multiple_tasks_sequential(self):
        """Test that tasks execute in order."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        order = []

        async def task(n):
            order.append(n)
            await asyncio.sleep(0.01)  # Small delay

        # Enqueue tasks
        for i in range(5):
            queue.enqueue(task, i)

        await queue.wait_for_completion()

        # Tasks should execute in FIFO order
        assert order == [0, 1, 2, 3, 4]

        # Cleanup
        await queue.stop_worker()

    async def test_wait_for_completion_empty_queue(self):
        """Test waiting for completion when queue is empty."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        # Should return immediately
        await queue.wait_for_completion()

        assert queue.queue.empty()

        # Cleanup
        await queue.stop_worker()

    async def test_task_with_async_operations(self):
        """Test tasks that perform async operations."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        results = []

        async def async_task(value):
            await asyncio.sleep(0.01)
            results.append(value * 2)

        queue.enqueue(async_task, 5)
        queue.enqueue(async_task, 10)

        await queue.wait_for_completion()

        assert 10 in results
        assert 20 in results

        # Cleanup
        await queue.stop_worker()

    async def test_graceful_shutdown_with_pending_tasks(self):
        """Test graceful shutdown clears pending tasks."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        executed = []

        async def slow_task(n):
            await asyncio.sleep(0.1)
            executed.append(n)

        # Enqueue tasks
        for i in range(3):
            queue.enqueue(slow_task, i)

        # Stop immediately (some tasks may not execute)
        await queue.stop_worker()

        assert queue._running is False

    async def test_worker_timeout_handling(self):
        """Test that worker handles queue timeout correctly."""
        queue = BackgroundTaskQueue()
        await queue.start_worker()

        # Let worker run with empty queue for a bit
        await asyncio.sleep(0.15)

        # Worker should still be running
        assert queue._running is True

        # Cleanup
        await queue.stop_worker()
