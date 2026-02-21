import pytest
from pytest_mock import MockerFixture
from typing import List, Any, Callable

from name_producer import SpeakerIdentificationProducer


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


def setup_rabbitmq_mocks(mocker: MockerFixture, message_bodies: List[str]):
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

    # Patch the connect_robust function to return our mock connection
    async def mock_connect_robust(*_args, **_kwargs):
        return mock_connection
    mocker.patch("name_producer.aio_pika.connect_robust", mock_connect_robust)
    
    # Run the producer
    producer = SpeakerIdentificationProducer(config)
    await producer.run()
    assert mock_channel.default_exchange.publish.called_once()
    called_body = mock_channel.default_exchange.publish.call_args_list[0].args[0].body.decode()
    assert called_body == """[0.00-5.00] SPEAKER_01:
this is what SPEAKER_01 said at time index 0

[5.00-10.00] SPEAKER_01:
this is what SPEAKER_01 said at time index 5

"""
    assert mock_channel.default_exchange.publish.call_args_list[0].kwargs == {"routing_key": "mockdestinationqueue"}
    assert producer.prompts_by_filename["filename1"] == ""
