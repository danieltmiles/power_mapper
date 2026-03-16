# Nomad client configuration for Apple Silicon (MPS) nodes.
#
# GPU-intensive jobs (llm, dads, whisper) are constrained to nodes where
# meta.gpu_type is set. Set it here to "mps" so the scheduler can target
# this node for GPU work.
#
# Start with: nomad agent -config=nomad.d/client-mps.hcl

datacenter = "dc1"
data_dir   = "/opt/nomad/data"
log_level  = "INFO"
bind_addr  = "0.0.0.0"

client {
  enabled = true

  # Replace with your Nomad server address(es).
  servers = ["<NOMAD_SERVER_ADDR>:4647"]

  meta {
    gpu_type = "mps"
    platform = "darwin"
  }
}

# raw_exec is used to run Python scripts directly via the venv interpreter.
plugin "raw_exec" {
  config {
    enabled = true
  }
}
