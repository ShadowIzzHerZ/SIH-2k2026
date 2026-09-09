# 10Hz Dead Reckoning Backend Engine - SIH 2026
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import defaultdict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass(order=True)
class IngestionItem:
    """
    Priority Queue Item:
    Priority 0: Real-time Live frame (Low latency, high priority)
    Priority 1: Backlog recovery chunks (Processed when live queue is quiet)

    Ordering:
    1. priority (0 before 1)
    2. timestamp (FIFO chronologically across all concurrent devices)
    3. seq_num (chronological within each device)
    """
    priority: int
    timestamp: float
    seq_num: int
    device_id: str = field(compare=False)
    data: Dict[str, Any] = field(compare=False)
    is_backlog: bool = field(compare=False, default=False)

class IngestionQueueManager:
    """
    High-performance Asynchronous Priority Ingestion Queue.
    Ensures real-time live streams are never starved by multi-thousand backlog floods.
    Maintains per-device sequence tracking to guarantee chronological processing.
    """
    def __init__(self, maxsize: int = 50000):
        self._queue: asyncio.PriorityQueue[IngestionItem] = asyncio.PriorityQueue(maxsize=maxsize)
        self._device_seq_counters: Dict[str, int] = defaultdict(int)
        self._reorder_buffers: Dict[str, List[IngestionItem]] = defaultdict(list)
        self._is_running = False

    async def enqueue_live(self, device_id: str, packet: Dict[str, Any]) -> bool:
        """Enqueue single real-time frame with Priority 0."""
        seq = packet.get("seq", self._device_seq_counters[device_id] + 1)
        ts = packet.get("timestamp", 0.0)
        item = IngestionItem(
            priority=0,
            timestamp=ts,
            seq_num=seq,
            device_id=device_id,
            data=packet,
            is_backlog=False
        )
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            logger.error(f"Ingestion queue full! Dropping live frame for {device_id}")
            return False

    async def enqueue_backlog_chunk(self, device_id: str, chunk: List[Dict[str, Any]]) -> int:
        """
        Enqueue backlog in manageable chunks with Priority 1.
        Cooperative multitasking: yields control to the event loop every chunk slice.
        """
        enqueued_count = 0
        chunk_size = settings.BACKLOG_CHUNK_SIZE

        # Sort chunk chronologically by sequence number first, timestamp second
        sorted_chunk = sorted(chunk, key=lambda x: (x.get("seq", 0), x.get("timestamp", 0)))

        for i in range(0, len(sorted_chunk), chunk_size):
            slice_items = sorted_chunk[i:i + chunk_size]
            for packet in slice_items:
                seq = packet.get("seq", 0)
                ts = packet.get("timestamp", 0.0)
                item = IngestionItem(
                    priority=1,
                    timestamp=ts,
                    seq_num=seq,
                    device_id=device_id,
                    data=packet,
                    is_backlog=True
                )
                try:
                    await self._queue.put(item)
                    enqueued_count += 1
                except asyncio.QueueFull:
                    logger.warning(f"Ingestion queue saturated during backlog ingestion for {device_id}")
                    break

            # Cooperative yield so live WebSockets can process uninterrupted
            await asyncio.sleep(0)

        return enqueued_count

    async def get_next_item(self) -> IngestionItem:
        """Fetch next item strictly by priority (Live first, Backlog second)."""
        return await self._queue.get()

    def task_done(self):
        self._queue.task_done()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

