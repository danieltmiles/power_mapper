# Services scheduled on the NVIDIA CUDA node.
#
# Runs the LLM inference worker and one Whisper worker (CPU mode).
# Both pull from the same shared queues as their MPS counterparts,
# providing additional throughput capacity.
#
# Submit: nomad job run nomad.d/jobs/cuda.nomad.hcl
#
# Secrets — populate once per cluster:
#   nomad var put nomad/jobs/power-mapper \
#     rabbitmq_password=... \
#     solr_password=... \
#     webdav_password=... \
#     model_path=/mnt/bfd/models/Qwen3-32B-Q4_K_M.gguf

job "power-mapper-cuda" {
  datacenters = ["dc1"]
  type        = "service"

  constraint {
    attribute = "${meta.gpu_type}"
    value     = "cuda"
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
          "cd /home/dmiles/power_mapper && source venv/bin/activate && exec python llm.py ${NOMAD_TASK_DIR}/config.json",
        ]
      }

      template {
        destination = "${NOMAD_TASK_DIR}/config.json"
        change_mode = "restart"
        data        = <<EOF
{{ with nomadVar "nomad/jobs/power-mapper" }}
{
  "work_queue":    "llm/qwen32",
  "model_path":    "{{ .model_path }}",
  "hf_model_name": "Qwen/Qwen3-32B",
  "host":     "rabbitmq.doodledome.org",
  "port":     5671,
  "username": "client",
  "password": "{{ .rabbitmq_password }}",
  "redis": {
    "host":         "doodledome.org",
    "port":         6380,
    "ssl":          true,
    "ssl_ca_certs": "/home/dmiles/power_mapper/tls/ca.crt",
    "ssl_certfile": "/home/dmiles/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/home/dmiles/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      resources {
        cpu    = 4000
        memory = 2000
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Whisper transcription — one CPU worker
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
          "cd /home/dmiles/power_mapper && source venv/bin/activate && exec python whisper_transcription.py ${NOMAD_TASK_DIR}/config.json --device=cpu",
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
    "ssl_ca_certs": "/home/dmiles/power_mapper/tls/ca.crt",
    "ssl_certfile": "/home/dmiles/power_mapper/tls/client.crt",
    "ssl_keyfile":  "/home/dmiles/power_mapper/tls/client.key"
  }
}
{{ end }}
EOF
      }

      env {
        OMP_NUM_THREADS = "4"
        MKL_NUM_THREADS = "4"
      }

      resources {
        cpu    = 10000
        memory = 5000
      }

      logs {
        max_files     = 5
        max_file_size = 15
      }
    }
  }
}
