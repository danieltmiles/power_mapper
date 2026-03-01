# Power Mapper Pipeline

Takes unstructured data, uses AI pipeline to make it usable.

## Goals
I have two co-equal goals in developing this code: Personal education/development, and that I hope it will be
genuinely useful. Please look kindly on the code, I'm developing it on my own in my spare time. I understand
there are standards I'm not following, such as an OpenAI-compatible web service for LLM requests, or even Ollama
Server. I've chosen to roll my own code for those things, not because it's the right technical dexision but
because it helps me educate myself. It is also a project in very early stages.

Goal: Take in unstructured data like YouTube videos of city council meetings, news articles, congressional voting
records, speeches, and anything else we can think to feed it. Produce a, "power map." -- A graph in a graphing
database of which people care about which issues, do they support or oppose the issues, what are their connections,
both to other people and to organizations?

Ideally, this will help answer questions like, "I care about Issue X, how do I get involved?"

## System Architecture

```mermaid
flowchart TD
    Start([Download YouTube Video]) --> Audio[Audio File MP3]
    
    Audio --> MINT[MINT Service<br/>Metadata Inference for<br/>Named Transcripts]
    MINT --> |Filename, Date, Title| DADS[DADS Service<br/>Diarization and<br/>Detection Service]
    
    DADS --> |Speaker Segments<br/>+ Metadata| SLICE[SLICE Service<br/>Segment and Label<br/>Individual Chunks for Extraction]
    SLICE --> |Segmented Audio<br/>+ Metadata| Whisper[Whisper Transcription]
    
    Whisper --> |Raw Transcription<br/>+ Speaker Labels<br/>+ Start/End Times| CLEAN[CLEAN Service<br/>Contextual Language Error<br/>Analysis and Normalization]
    
    CLEAN --> |Cleaned Transcription<br/>+ All Metadata| GATE[GATE Service<br/>Guess Assessment and<br/>Transcription Evaluation]
    
    GATE --> |Rejected| Queue1[Requeue for<br/>Re-processing]
    GATE --> |Accepted| DB1[(Database<br/>Segments)]
    GATE --> |Accepted| GateSplit{Split Topology<br/>Topic Exchange}
    
    GateSplit --> NAME[NAME Service<br/>Natural Attribution from<br/>Mentioned Entities]
    NAME --> |Speaker Names<br/>+ Confidence Scores| DB2[(Database<br/>Update Speakers)]
    GateSplit --> Topics[Get Topics Service]
    GateSplit --> People[Get People Service]
    GateSplit --> Orgs[Get Organizations Service]
    
    Topics --> |Topics + Descriptions| DB3[(Database<br/>Topics)]
    People --> |People + Descriptions| DB4[(Database<br/>People)]
    Orgs --> |Organizations + Descriptions| DB5[(Database<br/>Organizations)]
    
    Queue1 -.-> CLEAN
    
    style MINT fill:#000,stroke:#fff,color:#fff
    style DADS fill:#000,stroke:#fff,color:#fff
    style SLICE fill:#000,stroke:#fff,color:#fff
    style Whisper fill:#000,stroke:#fff,color:#fff
    style CLEAN fill:#000,stroke:#fff,color:#fff
    style GATE fill:#000,stroke:#fff,color:#fff
    style NAME fill:#000,stroke:#fff,color:#fff
    style Topics fill:#000,stroke:#fff,color:#fff
    style People fill:#000,stroke:#fff,color:#fff
    style Orgs fill:#000,stroke:#fff,color:#fff
    style DB1 fill:#000,stroke:#fff,color:#fff
    style DB2 fill:#000,stroke:#fff,color:#fff
    style DB3 fill:#000,stroke:#fff,color:#fff
    style DB4 fill:#000,stroke:#fff,color:#fff
    style DB5 fill:#000,stroke:#fff,color:#fff
```

## Pipeline Stages

### 1. Download Stage
Downloads YouTube videos in audio-only mode using `yt-dlp`:
```bash
yt-dlp -f "bestaudio" -x --audio-format mp3 --dateafter now-15years --sleep-interval 3600 "https://www.youtube.com/@egovpdx8714/videos"
```

### 2. MINT Service
**Metadata Inference for Named Transcripts**
- Extracts metadata from filename
- Infers date and title information

### 3. DADS Service
**Diarization and Detection Service**
- Processes audio for speaker diarization
- Produces speaker segments with timing information
- Publishes to RabbitMQ queue

### 4. Whisper Transcription
- Transcribes audio segments
- Preserves metadata:
  - Filename, date, title
  - Start/end times
  - Speaker labels

### 5. CLEAN Service
**Contextual Language Error Analysis and Normalization**
- AI-powered cleanup of transcriptions
- Preserves all metadata including original transcription

### 6. GATE Service
**Guess Assessment and Transcription Evaluation**
- Quality checks cleaned transcriptions
- Accepts or rejects segments
- Accepted segments stored in database
- Rejected segments requeued

### 7. NAME Service
**Natural Attribution from Mentioned Entities**
- Speaker identification from transcript bundles
- Updates database with speaker names and confidence scores
- Uses sliding window approach

### 8. Parallel Analysis (Post-Cut Line)
Three parallel services process bundled transcriptions:

#### Get Topics Service
- Extracts topics and descriptions
- Stores in database

#### Get People Service
- Identifies people mentioned
- Stores with descriptions

#### Get Organizations Service
- Identifies organizations mentioned
- Stores with descriptions

## Message Flow
All services communicate via RabbitMQ queues, preserving metadata throughout the pipeline:
- Filename
- Date
- Title
- Start/End times
- Speaker labels
- Original/raw transcription
- Confidence scores
