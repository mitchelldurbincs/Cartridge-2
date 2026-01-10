# Cluster outputs
output "cluster_name" {
  description = "GKE cluster name"
  value       = module.gke.cluster_name
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = module.gke.cluster_location
}

output "get_credentials_command" {
  description = "Command to get cluster credentials"
  value       = "gcloud container clusters get-credentials ${module.gke.cluster_name} --region ${module.gke.cluster_location} --project ${var.project_id}"
}

# Database outputs
output "database_connection_name" {
  description = "Cloud SQL connection name (for Cloud SQL Proxy)"
  value       = module.cloud_sql.instance_connection_name
}

output "database_private_ip" {
  description = "Cloud SQL private IP"
  value       = module.cloud_sql.private_ip_address
  sensitive   = true
}

output "database_password" {
  description = "Database password"
  value       = random_password.db_password.result
  sensitive   = true
}

# Storage outputs
output "models_bucket" {
  description = "GCS bucket for models"
  value       = module.storage.models_bucket_name
}

output "docker_registry" {
  description = "Docker registry URL"
  value       = module.storage.docker_registry_url
}

# IAM outputs
output "service_account_annotations" {
  description = "Annotations for Kubernetes service accounts"
  value       = module.iam.k8s_sa_annotations
}

# Helper outputs for configuration
output "cartridge_env_vars" {
  description = "Environment variables for Cartridge workloads"
  sensitive   = true
  value       = <<-EOT
    # Database connection (use with Cloud SQL Proxy sidecar)
    CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:${random_password.db_password.result}@127.0.0.1:5432/cartridge

    # Or direct connection via private IP
    CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:${random_password.db_password.result}@${module.cloud_sql.private_ip_address}:5432/cartridge

    # GCS model storage (S3-compatible API)
    CARTRIDGE_STORAGE_MODEL_BACKEND=s3
    CARTRIDGE_STORAGE_S3_BUCKET=${module.storage.models_bucket_name}
    CARTRIDGE_STORAGE_S3_ENDPOINT=https://storage.googleapis.com
  EOT
}
