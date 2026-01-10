# Storage Module - GCS Bucket and Artifact Registry
# Creates storage resources for models and container images

# GCS Bucket for model storage
resource "google_storage_bucket" "models" {
  name          = "${var.project_id}-${var.name_prefix}-models"
  project       = var.project_id
  location      = var.region
  storage_class = "STANDARD"

  # Uniform bucket-level access (recommended)
  uniform_bucket_level_access = true

  # Enable versioning to keep model history
  versioning {
    enabled = var.enable_versioning
  }

  # Lifecycle rules to manage old versions
  dynamic "lifecycle_rule" {
    for_each = var.enable_versioning ? [1] : []
    content {
      condition {
        num_newer_versions = var.versions_to_keep
      }
      action {
        type = "Delete"
      }
    }
  }

  # Delete incomplete multipart uploads
  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }

  # Prevent accidental deletion
  force_destroy = !var.deletion_protection

  labels = var.labels
}

# Artifact Registry for container images
resource "google_artifact_registry_repository" "images" {
  repository_id = "${var.name_prefix}-images"
  project       = var.project_id
  location      = var.region
  format        = "DOCKER"
  description   = "Container images for Cartridge2"

  # Cleanup policies
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.images_to_keep
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  labels = var.labels
}
