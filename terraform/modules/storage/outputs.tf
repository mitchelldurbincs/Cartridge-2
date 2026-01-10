output "models_bucket_name" {
  description = "GCS bucket name for models"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "GCS bucket URL for models"
  value       = google_storage_bucket.models.url
}

output "models_bucket_self_link" {
  description = "GCS bucket self link"
  value       = google_storage_bucket.models.self_link
}

output "artifact_registry_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.images.id
}

output "artifact_registry_name" {
  description = "Artifact Registry repository name"
  value       = google_artifact_registry_repository.images.name
}

output "docker_registry_url" {
  description = "Docker registry URL for pushing images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
