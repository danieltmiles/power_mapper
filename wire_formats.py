import datetime
from dataclasses import dataclass, asdict, field
from typing import Protocol, Any
from serialization import dumps, load


class Dictable(Protocol):
    def asdict(self) -> dict:
        ...


@dataclass
class Metaparams(Dictable):
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.2

    def asdict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty
        }


@dataclass
class LLMPromptJob(Dictable):
    job_id: str
    filename: str
    reply_to: str
    prompt: str
    request_id: str | None = None
    meta_params: Metaparams = field(default_factory=Metaparams)

    def __init__(
        self,
        job_id: str,
        filename: str,
        reply_to: str,
        prompt: str,
        request_id: str | None = None,
        meta_params: Metaparams | dict[str, Any] | None = None
    ):
        self.job_id = job_id
        self.filename = filename
        self.reply_to = reply_to
        self.prompt = prompt
        self.request_id = request_id
        if meta_params:
            if isinstance(meta_params, dict):
                self.meta_params = Metaparams(**meta_params)
            elif isinstance(meta_params, Metaparams):
                self.meta_params = meta_params
            else:
                raise ValueError("meta_params must be a Metaparams instance or a dict")
        else:
            self.meta_params = Metaparams()



    def asdict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "reply_to": self.reply_to,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "meta_params": self.meta_params.asdict() if self.meta_params else None
        }

@dataclass
class LLMPromptResponse(LLMPromptJob):
    generated_text: str | None = None

    @classmethod
    def from_llm_prompt_job(cls, job: LLMPromptJob, generated_text: str | None = None) -> 'LLMPromptResponse':
        return LLMPromptResponse(
            job_id=job.job_id,
            filename=job.filename,
            reply_to=job.reply_to,
            prompt=job.prompt,
            request_id=job.request_id,
            meta_params=job.meta_params,
            generated_text=generated_text
        )
    def asdict(self) -> dict[str, Any]:
        base_dict = super().asdict()
        base_dict["generated_text"] = self.generated_text
        return base_dict

@dataclass
class TranscriptMetadata(Dictable):
    filename: str
    meeting_title: str | None
    session_type: str | None
    date: datetime.datetime | None
    video_id: str | None

    def __init__(
            self,
            filename: str,
            meeting_title: str | None = None,
            session_type: str | None = None,
            date: datetime.datetime | str | None = None,
            video_id: str | None = None,
    ):
        self.filename = filename
        self.meeting_title = meeting_title
        self.session_type = session_type
        if isinstance(date, str):
            try:
                self.date = datetime.datetime.fromisoformat(date)
            except ValueError:
                self.date = None
        else:
            self.date = date
        self.video_id = video_id

    def asdict(self):
        return {
            "filename": self.filename,
            "meeting_title": self.meeting_title,
            "session_type": self.session_type,
            "date": self.date.isoformat() if self.date else None,
            "video_id": self.video_id
        }

@dataclass
class DiarizationResponse(Dictable):
    diarization: str
    transcript_metadata: TranscriptMetadata

    def asdict(self) -> dict[str, Any]:
        return {
            "transcript_metadata": self.transcript_metadata.asdict(),
            "diarization_data": self.diarization,
        }