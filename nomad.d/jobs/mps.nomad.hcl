# All services scheduled on the Apple Silicon (MPS) node.
#
# Includes one instance of every service except whisper, plus two whisper
# workers running on CPU (--device=cpu) to avoid contending with MPS-based
# LLM and diarization work.
#
# Submit: nomad job run nomad.d/jobs/mps.nomad.hcl
#
# Secrets — populate once per cluster:
#   nomad var put nomad/jobs/power-mapper \
#     rabbitmq_password=... \
#     solr_password=... \
#     webdav_password=... \
#     model_path_mps=/path/to/Qwen3-32B-Q4_K_M.gguf

job "power-mapper-mps" {
  datacenters = ["dc1"]
  type        = "service"

  constraint {
    attribute = "${meta.gpu_type}"
    value     = "mps"
  }

  # ---------------------------------------------------------------------------
  # LLM inference worker
  # ---------------------------------------------------------------------------
  group "llm" {
    count = 1

    restart {
      attempts = 5
      interval = "10m"
      delay    = "30s"
      mode     = "delay"
    }

    task "llm" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec caffeinate python llm.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":    "llm/qwen32",
  "model_path":    "{{ .model_path_mps }}",
  "hf_model_name": "Qwen/Qwen3-32B",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 4000
        memory = 28000
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Whisper transcription — two CPU workers
  # ---------------------------------------------------------------------------
  group "whisper" {
    count = 1

    restart {
      attempts = 5
      interval = "10m"
      delay    = "30s"
      mode     = "delay"
    }

    task "whisper" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python whisper_transcription.py ${NOMAD_TASK_DIR}/config.json --device=mps",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":    "whispers",
  "results_queue": "transcriptions",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 4000
        memory = 5000
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Mint — filename → TranscriptMetadata
  # ---------------------------------------------------------------------------
  group "mint" {
    count = 1

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "mint" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python mint.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":   "filenames",
  "result_queue": "transcript-file-details",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # DADS — speaker diarization
  # ---------------------------------------------------------------------------
  group "dads" {
    count = 1

    restart {
      attempts = 5
      interval = "10m"
      delay    = "30s"
      mode     = "delay"
    }

    task "dads" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin && cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python dads.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "host":              "rabbitmq.doodledome.org",
  "port":              5671,
  "username":          "client",
  "password":          "{{ .rabbitmq_password }}",
  "work_queue":        "transcript-file-details",
  "destination_queue": "diarizations",
  "storage_info": {
    "type":     "webdav",
    "url":      "https://webdav.doodledome.org/",
    "username": "dmiles",
    "password": "{{ .webdav_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 2000
        memory = 6000
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Slice — diarization → Whisper segments
  # ---------------------------------------------------------------------------
  group "slice" {
    count = 1

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "slice" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin && cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python slice.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "host":              "rabbitmq.doodledome.org",
  "port":              5671,
  "username":          "client",
  "password":          "{{ .rabbitmq_password }}",
  "work_queue":        "diarizations",
  "destination_queue": "whispers",
  "storage_info": {
    "type":     "webdav",
    "url":      "https://webdav.doodledome.org/",
    "username": "dmiles",
    "password": "{{ .webdav_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Clean — LLM-powered transcript cleanup
  # ---------------------------------------------------------------------------
  group "clean" {
    count = 0

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "clean" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python clean.py --concurrent=5 ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":    "transcriptions",
  "results_queue": "cleaned-transcriptions",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Gate — quality check; routes pass/retry
  # ---------------------------------------------------------------------------
  group "gate" {
    count = 0

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "gate" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python gate.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":            "cleaned-transcriptions",
  "accepted_queue":        "cleaned-transcriptions-accepted",
  "retry_queue":           "transcriptions",
  "max_tries":             3,
  "hf_model_name":         "Qwen/Qwen3-32B",
  "embedding_model_path":  "/Users/dmiles/Qwen3-Embedding-8B-Q4_K_M.gguf",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "solr": {
    "url":      "https://solr.doodledome.org/solr",
    "username": "content-user",
    "password": "{{ .solr_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Name producer — sliding window speaker identification
  # ---------------------------------------------------------------------------
  group "name-producer" {
    count = 0

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "name-producer" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python name_producer.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":        "cleaned-transcriptions-accepted",
  "destination_queue": "llm/qwen32",
  "reply_to":          "identifications",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "solr": {
    "url":      "https://solr.doodledome.org/solr",
    "username": "content-user",
    "password": "{{ .solr_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Name consumer — processes identification results
  # ---------------------------------------------------------------------------
  group "name-consumer" {
    count = 0

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "name-consumer" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python name_consumer.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":        "identifications",
  "destination_queue": "trac",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "solr": {
    "url":      "https://solr.doodledome.org/solr",
    "username": "content-user",
    "password": "{{ .solr_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Trac — topic, relationship, and context extraction
  # ---------------------------------------------------------------------------
  group "trac" {
    count = 0

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "trac" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python trac.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue": "trac",
  "reply_to":   "person-issue-relationships",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "solr": {
    "url":      "https://solr.doodledome.org/solr",
    "username": "content-user",
    "password": "{{ .solr_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Scribe producer — generates issue descriptions
  # ---------------------------------------------------------------------------
  group "scribe-producer" {
    count = 0

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "scribe-producer" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python scribe.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue": "person-issue-relationships",
  "reply_to":   "issue-descriptions",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "solr": {
    "url":      "https://solr.doodledome.org/solr",
    "username": "content-user",
    "password": "{{ .solr_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Scribe consumer — persists issue descriptions to Solr
  # ---------------------------------------------------------------------------
  group "scribe-consumer" {
    count = 0

    restart {
      attempts = 5
      interval = "5m"
      delay    = "15s"
      mode     = "delay"
    }

    task "scribe-consumer" {
      driver = "raw_exec"

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/dmiles/code/power_mapper && source venv/bin/activate && exec python scribe_consumer.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":           "issue-descriptions",
  "embedding_model_path": "/Users/dmiles/Qwen3-Embedding-8B-Q4_K_M.gguf",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "solr": {
    "url":      "https://solr.doodledome.org/solr",
    "username": "content-user",
    "password": "{{ .solr_password }}"
  },
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/Users/dmiles/code/power_mapper/tls/ca.crt",
    "ssl_certfile": "/Users/dmiles/code/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/Users/dmiles/code/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 500
        memory = 512
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

}
