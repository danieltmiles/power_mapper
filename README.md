# Transcribinator Pipeline

A comprehensive audio transcription and analysis pipeline for processing YouTube videos through multiple AI services.

## System Architecture

```mermaid
flowchart TD
    Start([Download YouTube Video]) --> Audio[Audio File MP3]
    
    Audio --> MINT[MINT Service<br/>Metadata Inference for<br/>Named Transcripts]
    MINT --> |Filename, Date, Title| DADS[DADS Service<br/>Diarization and<br/>Detection Service]
    
    DADS --> |Speaker Segments<br/>+ Metadata| Whisper[Whisper Transcription]
    
    Whisper --> |Raw Transcription<br/>+ Speaker Labels<br/>+ Start/End Times| CLEAN[CLEAN Service<br/>Contextual Language Error<br/>Analysis and Normalization]
    
    CLEAN --> |Cleaned Transcription<br/>+ All Metadata| GATE[GATE Service<br/>Guess Assessment and<br/>Transcription Evaluation]
    
    GATE --> |Rejected| Queue1[Requeue for<br/>Re-processing]
    GATE --> |Accepted| DB1[(Database<br/>Segments)]
    
    DB1 --> Bundle[Bundle/Sliding Window<br/>Aggregation]
    
    Bundle --> NAME[NAME Service<br/>Natural Attribution from<br/>Mentioned Entities]
    
    NAME --> |Speaker Names<br/>+ Confidence Scores| DB2[(Database<br/>Update Speakers)]
    
    Bundle --> Split{Split Topology}
    
    Split --> Topics[Get Topics Service]
    Split --> People[Get People Service]
    Split --> Orgs[Get Organizations Service]
    
    Topics --> |Topics + Descriptions| DB3[(Database<br/>Topics)]
    People --> |People + Descriptions| DB4[(Database<br/>People)]
    Orgs --> |Organizations + Descriptions| DB5[(Database<br/>Organizations)]
    
    Queue1 -.-> CLEAN
    
    style MINT fill:#e1f5ff
    style DADS fill:#e1f5ff
    style CLEAN fill:#e1f5ff
    style GATE fill:#e1f5ff
    style NAME fill:#e1f5ff
    style Topics fill:#ffe1f5
    style People fill:#ffe1f5
    style Orgs fill:#ffe1f5
    style DB1 fill:#fff4e1
    style DB2 fill:#fff4e1
    style DB3 fill:#fff4e1
    style DB4 fill:#fff4e1
    style DB5 fill:#fff4e1
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
