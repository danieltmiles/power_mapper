# Nomad server configuration for the Power Mapper cluster.
# Run on one node (or an odd number for HA — set bootstrap_expect accordingly).
#
# Start with: nomad agent -config=nomad.d/server.hcl

datacenter = "dc1"
data_dir   = "/mnt/bfd/nomad/data"
log_level  = "INFO"
bind_addr  = "0.0.0.0"

server {
  enabled          = true
  bootstrap_expect = 1

  # Give GPU workloads more time to heartbeat before being marked lost.
  heartbeat_grace = "30s"
}

addresses {
  http = "0.0.0.0"
  rpc  = "0.0.0.0"
  serf = "0.0.0.0"
}

ports {
  http = 4646
  rpc  = 4647
  serf = 4648
}

