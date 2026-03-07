import hashlib
import pickle
from asyncio import Semaphore
from contextlib import asynccontextmanager
from typing import Protocol, Iterator, Any

import aiormq
import redis.asyncio as redis
from aio_pika.abc import AbstractRobustConnection, AbstractIncomingMessage, AbstractChannel
from aiormq import ChannelInvalidStateError

from utils import dial_rabbit_from_config, publish_event


class CachedIncomingMessage(AbstractIncomingMessage):
    # TODO: maybe in the future this will need to cache other things, like headers or what not
    def __init__(self, body: str | bytes):
        if isinstance(body, str):
            self.body = body.encode("utf-8")
        else:
            self.body = body

    @property
    def channel(self) -> aiormq.abc.AbstractChannel:
        pass

    def process(self, requeue: bool = False, reject_on_redelivered: bool = False,
                ignore_processed: bool = False) -> "AbstractProcessContext":
        pass

    async def ack(self, multiple: bool = False) -> None:
        pass

    async def reject(self, requeue: bool = False) -> None:
        pass

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None:
        pass

    @property
    def processed(self) -> bool:
        pass

    @property
    def locked(self) -> bool:
        pass

    @property
    def properties(self) -> aiormq.spec.Basic.Properties:
        pass

    def __iter__(self) -> Iterator[int]:
        pass

    def lock(self) -> None:
        pass


class CachedMessageIterator:
    def __init__(
        self,
        rabbitmq_connection: AbstractRobustConnection,
        redis_client: redis.Redis,
        queue_name: str,
        redis_key_prefix: str,
        redis_concurrency_limit: int = 10,
        config: dict[str, Any] = None,
    ):
        self.rabbitmq_connection = rabbitmq_connection
        self.redis_client = redis_client
        self.queue_name = queue_name
        self.redis_key_prefix = redis_key_prefix
        self.redis_concurrency_limit = redis_concurrency_limit
        self.redis_semaphore = Semaphore(self.redis_concurrency_limit)
        self.cached_messages = []
        self.active_message_keys_by_body_hash = {}
        self.config = config

    async def __aexit__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        pattern = f"{self.redis_key_prefix}*"
        cursor = 0
        all_keys = []

        async with self.redis_semaphore:
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                all_keys.extend(keys)
                if cursor == 0:
                    break
        if all_keys:
            print(f"Found {len(all_keys)} backed up messages in Redis, recovering...")
            recovered_count = 0

            # Process each backed up message using our existing channel
            for key in all_keys:
                # Get the message body from Redis
                async with self.redis_semaphore:
                    message_body = await self.redis_client.get(key)
                    self.active_message_keys_by_body_hash[self.md5sum(message_body)] = key
                    self.cached_messages.append(message_body)

            if self.config:
                await publish_event(
                    self.config,
                    f"REDIS_RECOVERY: [{self.redis_key_prefix}] queue={self.queue_name} "
                    f"replaying {len(all_keys)} messages from Redis backup. "
                    f"These were previously acked from RabbitMQ but not marked as fully processed. "
                    f"If downstream side-effects already occurred, this will cause duplicate processing."
                )

        await self.establish_channel_and_queue()
        return self

    async def establish_channel_and_queue(self):
        self.channel = await self.rabbitmq_connection.channel()
        await self.channel.set_qos(prefetch_count=1)
        self.queue = await self.channel.declare_queue(self.queue_name, durable=True)

    @classmethod
    def md5sum(cls, s: str | bytes) -> str:
        s_bytes = s if isinstance(s, bytes) else s.encode("utf-8")
        return hashlib.md5(s_bytes).hexdigest()

    def _get_redis_key(self, message: AbstractIncomingMessage | CachedIncomingMessage | str | bytes) -> str:
        if hasattr(message, "message_id"):
            return f"{self.redis_key_prefix}:{message.message_id}"
        if isinstance(message, CachedIncomingMessage):
            message = message.body
        return self.active_message_keys_by_body_hash.pop(self.md5sum(message))

    async def __aiter__(self):
        for message in self.cached_messages:
            body = message if isinstance(message, bytes) else message.encode("utf-8")
            yield CachedIncomingMessage(body=body)
        while True:
            # Explicitly set no_ack=False to ensure manual acknowledgment mode
            async with self.queue.iterator(no_ack=False) as queue_iter:
                async for message in queue_iter:
                    try:
                        redis_key = self._get_redis_key(message)
                        body = message.body.decode()
                        self.active_message_keys_by_body_hash[self.md5sum(body)] = redis_key
                        await self.redis_client.set(redis_key, body)
                        await message.ack()
                        yield message
                    except ChannelInvalidStateError as cise:
                        print(f"Rabbitmq Invalid State: {cise}, redialing...")
                        if self.config:
                            await publish_event(
                                self.config,
                                f"CHANNEL_REDIAL: [{self.redis_key_prefix}] queue={self.queue_name} "
                                f"ChannelInvalidStateError during message consume/ack cycle: {cise}. "
                                f"Redialing connection. The in-flight message may be redelivered by RabbitMQ "
                                f"if it was not yet acked."
                            )
                        self.rabbitmq_connection = await dial_rabbit_from_config(self.config)
                        await self.establish_channel_and_queue()


    @asynccontextmanager
    async def processing(self, message: AbstractIncomingMessage):
        try:
            yield message
        finally:
            await self.mark_processed(message)

    async def mark_processed(self, message: AbstractIncomingMessage):
        await self.redis_client.delete(self._get_redis_key(message))
