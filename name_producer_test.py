import pytest
from pytest_mock import MockerFixture
from typing import Any, Callable

from message_generator import generate_messages
from name_producer import SpeakerIdentificationProducer, form_transcript_chunk
from utils import get_answer


@pytest.fixture
def config():
    return {
        "host": "mockhost",
        "port": 1234,
        "username": "mockusername",
        "password": "mockpassword",
        "work_queue": "mockworkqueue",
        "destination_queue": "mockdestinationqueue",
    }

@pytest.fixture
def make_serialization_identity_loader(mocker: MockerFixture):
    """Factory fixture that returns a function to create identity loaders for any module."""
    def _make_loader(host_module: str):
        return mocker.patch(f"{host_module}.serialization.load", lambda x: x)
    return _make_loader

@pytest.fixture
def name_producer_identity_loader(make_serialization_identity_loader: Callable):
    make_serialization_identity_loader("name_producer")


def setup_rabbitmq_mocks(mocker: MockerFixture, message_bodies: list):
    # Create mock connection
    mock_connection = mocker.AsyncMock()
    mock_connection.__aenter__.return_value = mock_connection
    
    # Create mock channel
    mock_channel = mocker.AsyncMock()
    mock_connection.channel.return_value = mock_channel
    
    # Create mock queue
    mock_queue = mocker.MagicMock()
    mock_channel.declare_queue.return_value = mock_queue
    
    # Create mock messages from the provided message bodies
    mock_messages = []
    for body in message_bodies:
        mock_message = mocker.MagicMock()
        mock_message.body.decode.return_value = body
        mock_messages.append(mock_message)
    
    # Set up mock queue iterator with proper async protocol
    mock_queue_iterator = mocker.MagicMock()
    mock_queue_iterator.__aiter__ = mocker.MagicMock(return_value=mock_queue_iterator)
    mock_queue_iterator.__aenter__ = mocker.AsyncMock(return_value=mock_queue_iterator)
    mock_queue_iterator.__aexit__ = mocker.AsyncMock()
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
@pytest.mark.usefixtures("name_producer_identity_loader")
async def test_name_producer_reorders_out_of_order_messages(config: dict[str, Any], mocker: MockerFixture):

    # Set up mocks with two test messages
    mock_msg1 = mocker.MagicMock()
    mock_msg1.body.decode.return_value = mock_msg1
    mock_msg1.whisper_result.transcript_metadata.filename = "filename1"
    mock_msg1.whisper_result.segment_count = 1
    mock_msg1.whisper_result.total_segments = 2
    mock_msg1.whisper_result.speaker = "SPEAKER_01"
    mock_msg1.whisper_result.timings.start =  5
    mock_msg1.whisper_result.timings.end =  10
    mock_msg1.cleaned_transcript = "this is what SPEAKER_01 said at time index 5"

    mock_msg2 = mocker.MagicMock()
    mock_msg2.body.decode.return_value = mock_msg2
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
    assert """[0.00-5.00] SPEAKER_01:
this is what SPEAKER_01 said at time index 0

[5.00-10.00] SPEAKER_01:
this is what SPEAKER_01 said at time index 5""" in called_body
    assert mock_channel.default_exchange.publish.call_args_list[0].kwargs == {"routing_key": "mockdestinationqueue"}
    assert producer.whisper_results_by_filename["filename1"] == []

@pytest.mark.asyncio
@pytest.mark.usefixtures("name_producer_identity_loader")
async def test_finishes_sending(config: dict[str, Any], mocker: MockerFixture):
    msgs = generate_messages(num_messages=8, seed=42)
    prompt_text = "".join(form_transcript_chunk(r) for r in msgs)
    new_max = len(prompt_text) // 2 + len(prompt_text) // 6
    mocker.patch("name_producer.MAX_PROMPT_TRANSCRIPT_LENGTH", new_max)
    mock_connection, mock_channel = setup_rabbitmq_mocks(mocker, msgs)
    producer = SpeakerIdentificationProducer(config)
    await producer.run()
    assert mock_channel.default_exchange.publish.call_count == 3  # because of the sliding window
    last_sent_body: str = mock_channel.default_exchange.publish.call_args_list[2].args[0].body.decode()
    print(last_sent_body)
    assert last_sent_body.endswith("```\n")
    last_sent_body = last_sent_body[:len("```\n") * -1]
    last_sent_body = last_sent_body.strip()
    sorted_msgs = sorted(msgs, key=lambda x: x.whisper_result.segment_count)
    last_transcript_section = sorted_msgs[-1].cleaned_transcript.strip()
    assert last_sent_body.endswith(last_transcript_section)

def test_heap_peek():
    from name_producer import safe_heappeek
    import heapq
    heap = []
    heapq.heappush(heap, (2, "item 2"))
    heapq.heappush(heap, (1, "item 1"))
    peek = safe_heappeek(heap)
    assert peek == (1, "item 1")
    popped = heapq.heappop(heap)
    assert popped == (1, "item 1")

@pytest.mark.asyncio
async def test_solr_fetch():
    config = {
        "solr": {
            "url": "https://solr.doodledome.org/solr",
            "username": "content-user",
            "password": "QWVtkFmmH-4fRTCpj3mAbaVh"
        }
    }
    producer = SpeakerIdentificationProducer(config=config)
    fetched = await producer.fetch_from_solr_by_filename(filename="City Council  2015-03-04  AM [24T2kpG-9wE].mp3")
    print(fetched)