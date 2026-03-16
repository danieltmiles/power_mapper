# Nomad client configuration for NVIDIA CUDA nodes.
#
# Requires the nomad-device-nvidia plugin installed on the host:
#   https://developer.hashicorp.com/nomad/plugins/devices/nvidia
#
# GPU-intensive jobs (llm, dads, whisper) request `device "nvidia/gpu"` and
# are automatically scheduled here. CPU-only jobs run here too.
#
# Start with: nomad agent -config=nomad.d/client-cuda.hcl

datacenter = "dc1"
data_dir   = "/mnt/bfd/nomad/data"
log_level  = "INFO"
bind_addr  = "0.0.0.0"

client {
  enabled = true

  # Replace with your Nomad server address(es).
  servers = ["<NOMAD_SERVER_ADDR>:4647"]

  meta {
    gpu_type = "cuda"
    platform = "linux"
  }
}

plugin "raw_exec" {
  config {
    enabled = true
  }
}

# NVIDIA device plugin — exposes GPUs as schedulable resources.
plugin "nomad-device-nvidia" {
  config {
    enabled            = true
    ignored_gpu_ids    = []
    fingerprint_period = "1m"
  }
}
