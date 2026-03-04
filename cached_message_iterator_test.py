import pytest
from pytest_mock import MockerFixture

from cached_iterator import CachedMessageIterator


@pytest.mark.asyncio
async def test_cached_message_iterator(mocker: MockerFixture):
    # Create a proper mock message with the required attributes
    mock_message = mocker.AsyncMock()
    mock_message.message_id = "test-message-id"
    mock_message.body = b"hello this is rabbitmq message"
    mock_message.ack = mocker.AsyncMock()
    
    queue = mocker.MagicMock()
    queue.dmiles = "dmiles queue"
    # Use MagicMock for the iterator, not AsyncMock
    queue_iter = mocker.MagicMock()
    queue_iter.dmiles = "dmiles queue_iter"
    queue_iter.__aenter__ = mocker.AsyncMock(return_value=queue_iter)
    queue_iter.__aexit__ = mocker.AsyncMock(return_value=None)
    # Make __aiter__ return self explicitly using MagicMock
    queue_iter.__aiter__ = mocker.MagicMock(return_value=queue_iter)
    queue_iter.__anext__ = mocker.AsyncMock(side_effect=[mock_message, StopAsyncIteration])
    queue.iterator.return_value = queue_iter
    channel = mocker.AsyncMock()
    channel.dmiles = "dmiles channel"
    channel.declare_queue.return_value = queue
    connection = mocker.AsyncMock()
    connection.dmiles = "dmiles connection"
    connection.channel.return_value = channel
    redis_client = mocker.AsyncMock()
    redis_client.dmiles = "dmiles redis_client"
    mock_keys = ["backup:messages"]
    redis_client.scan.return_value = 0, mock_keys
    redis_client.get.return_value = "hello this is message"
    recvd_bodies = []
    async with CachedMessageIterator(
            rabbitmq_connection=connection,
            redis_client=redis_client,
            queue_name="my_queue",
            redis_key_prefix="backup:my_service",
            config={},
    ) as iterator:
        async for message in iterator:
            recvd_bodies.append(message.body.decode())

    assert recvd_bodies == [
        "hello this is message",
        "hello this is rabbitmq message"
    ]