import json

import pytest
from pytest_mock import MockerFixture
from typing import Any

from message_generator import generate_messages
from name_producer import SpeakerIdentificationProducer
from sliding_window import CleanedWhisperResultWrapper, SlidingWindow
from wire_formats import CleanedWhisperResult


@pytest.fixture
def config():
    return {
        "host": "mockhost",
        "port": 1234,
        "username": "mockusername",
        "password": "mockpassword",
        "work_queue": "mockworkqueue",
        "destination_queue": "mockdestinationqueue",
        "reply_to": "mockreplyto",
    }


def setup_rabbitmq_mocks(mocker: MockerFixture, message_bodies: list):
    # Create mock connection
    mock_connection = mocker.AsyncMock()
    mock_connection.__aenter__.return_value = mock_connection
    
    # Create mock channel
    mock_channel = mocker.AsyncMock()
    mock_connection.channel.return_value = mock_channel
    
    # Create mock queue
    mock_queue = mocker.MagicMock()
    mock_queue.dmiles = "mock_queue"
    mock_channel.declare_queue.return_value = mock_queue
    
    # Create mock messages from the provided message bodies
    mock_messages = []
    for body in message_bodies:
        mock_message = mocker.MagicMock()
        mock_message.body.decode.return_value = body
        mock_message.ack = mocker.AsyncMock()
        mock_messages.append(mock_message)
    
    # Set up mock queue iterator with proper async protocol
    mock_queue_iterator = mocker.MagicMock()
    mock_queue_iterator.dmiles = "mock_queue_iterator"
    mock_queue_iterator.__aiter__ = mocker.MagicMock(return_value=mock_queue_iterator)
    mock_queue_iterator.__aenter__ = mocker.AsyncMock(return_value=mock_queue_iterator)
    mock_queue_iterator.__aexit__ = mocker.AsyncMock(return_value=False)
    mock_queue_iterator.__anext__ = mocker.AsyncMock(
        side_effect=mock_messages + [StopAsyncIteration()]
    )
    mock_queue.iterator.return_value = mock_queue_iterator

    # Patch the connect_robust function to return our mock connection
    async def mock_connect_robust(*_args, **_kwargs):
        return mock_connection

    mocker.patch("name_producer.aio_pika.connect_robust", mock_connect_robust)

    return mock_connection, mock_channel


@pytest.mark.asyncio
async def test_name_producer_reorders_out_of_order_messages(config: dict[str, Any], mocker: MockerFixture):
    mocker.patch("name_producer.serialization.load", lambda x: x)

    # Set up mocks with two test messages
    mock_msg1 = mocker.MagicMock()
    mock_msg1.__class__ = CleanedWhisperResult
    mock_msg1.dmiles = "mock_msg1"
    mock_msg1.body.decode.return_value = mock_msg1
    mock_msg1.whisper_result.transcript_metadata.filename = "filename1"
    mock_msg1.whisper_result.segment_count = 1
    mock_msg1.whisper_result.total_segments = 2
    mock_msg1.whisper_result.speaker = "SPEAKER_01"
    mock_msg1.whisper_result.timings.start =  5
    mock_msg1.whisper_result.timings.end =  10
    mock_msg1.cleaned_transcript = "this is what SPEAKER_01 said at time index 5"

    mock_msg2 = mocker.MagicMock()
    mock_msg2.__class__ = CleanedWhisperResult
    mock_msg2.dmiles = "mock_msg2"
    mock_msg2.whisper_result.transcript_metadata.filename = "filename1"
    mock_msg2.whisper_result.segment_count = 0
    mock_msg2.whisper_result.total_segments = 2
    mock_msg2.whisper_result.speaker = "SPEAKER_01"
    mock_msg2.whisper_result.timings.start =  0
    mock_msg2.whisper_result.timings.end =  5
    mock_msg2.cleaned_transcript = "this is what SPEAKER_01 said at time index 0"

    mock_connection, mock_channel = setup_rabbitmq_mocks(mocker, [mock_msg1, mock_msg2])

    # Run the producer
    producer = SpeakerIdentificationProducer(config)
    await producer.run()
    assert mock_channel.default_exchange.publish.called_once()
    called_body = mock_channel.default_exchange.publish.call_args_list[0].args[0].body.decode()
    llm_prompt_job: dict[str, Any] = json.loads(called_body)
    assert """[0.00-5.00] SPEAKER_01:
this is what SPEAKER_01 said at time index 0

[5.00-10.00] SPEAKER_01:
this is what SPEAKER_01 said at time index 5""" in llm_prompt_job["prompt"]
    assert mock_channel.default_exchange.publish.call_args_list[0].kwargs == {"routing_key": "mockdestinationqueue"}


@pytest.mark.asyncio
async def test_finishes_sending(config: dict[str, Any], mocker: MockerFixture):
    mocker.patch("name_producer.serialization.load", lambda x: x)

    msgs = generate_messages(num_messages=8, seed=42)
    prompt_text = "".join(SlidingWindow.form_transcript_chunk(CleanedWhisperResultWrapper(r)) for r in msgs)
    new_max = len(prompt_text) // 2 + len(prompt_text) // 6
    mocker.patch("name_producer.MAX_PROMPT_TRANSCRIPT_LENGTH", new_max)
    mock_connection, mock_channel = setup_rabbitmq_mocks(mocker, msgs)
    producer = SpeakerIdentificationProducer(config)
    await producer.run()
    assert mock_channel.default_exchange.publish.call_count == 3  # because of the sliding window
