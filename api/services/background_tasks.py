"""Background task processing for async operations."""

import asyncio
import contextlib
from collections.abc import Callable

import structlog

logger = structlog.get_logger(__name__)


class BackgroundTaskQueue:
    """Simple in-memory task queue for background processing."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self._running = False

    async def start_worker(self):
        """Start background worker."""
        if self._running:
            logger.warning("background_worker_already_running")
            return

        self._running = True
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("background_worker_started")

    async def stop_worker(self):
        """Stop background worker."""
        if not self._running:
            return

        self._running = False

        if self.worker_task:
            self.worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker_task
            logger.info("background_worker_stopped")

    async def _worker(self):
        """Process tasks from queue."""
        while self._running:
            try:
                task_func, args, kwargs = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                try:
                    await task_func(*args, **kwargs)
                except Exception as e:
                    logger.error(
                        "background_task_execution_error",
                        task=task_func.__name__,
                        error=str(e),
                    )
                finally:
                    self.queue.task_done()
            except TimeoutError:
                # No tasks in queue, continue waiting
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("background_worker_error", error=str(e))

    def enqueue(self, task_func: Callable, *args, **kwargs):
        """Add task to queue.

        Args:
            task_func: Async function to execute
            *args: Positional arguments for task_func
            **kwargs: Keyword arguments for task_func
        """
        self.queue.put_nowait((task_func, args, kwargs))
        logger.debug("task_enqueued", task=task_func.__name__, queue_size=self.queue.qsize())

    async def wait_for_completion(self):
        """Wait for all queued tasks to complete."""
        await self.queue.join()


# Global task queue
task_queue = BackgroundTaskQueue()
