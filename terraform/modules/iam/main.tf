# IAM Module - Service Accounts and Workload Identity
# Creates service accounts with appropriate permissions for Cartridge2 workloads

# Service account for Actor pods
resource "google_service_account" "actor" {
  account_id   = "${var.name_prefix}-actor"
  project      = var.project_id
  display_name = "Cartridge Actor Service Account"
  description  = "Service account for Actor pods (self-play)"
}

# Service account for Trainer pods
resource "google_service_account" "trainer" {
  account_id   = "${var.name_prefix}-trainer"
  project      = var.project_id
  display_name = "Cartridge Trainer Service Account"
  description  = "Service account for Trainer pod (ML training)"
}

# Service account for Web/Frontend pods
resource "google_service_account" "web" {
  account_id   = "${var.name_prefix}-web"
  project      = var.project_id
  display_name = "Cartridge Web Service Account"
  description  = "Service account for Web and Frontend pods"
}

# Grant Actor access to GCS bucket (read models)
resource "google_storage_bucket_iam_member" "actor_gcs_reader" {
  bucket = var.models_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.actor.email}"
}

# Grant Trainer access to GCS bucket (read/write models)
resource "google_storage_bucket_iam_member" "trainer_gcs_writer" {
  bucket = var.models_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.trainer.email}"
}

# Grant Web access to GCS bucket (read models for serving)
resource "google_storage_bucket_iam_member" "web_gcs_reader" {
  bucket = var.models_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.web.email}"
}

# Cloud SQL Client role for database access
resource "google_project_iam_member" "actor_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.actor.email}"
}

resource "google_project_iam_member" "trainer_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_project_iam_member" "web_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.web.email}"
}

# Workload Identity bindings
# These allow Kubernetes service accounts to impersonate GCP service accounts

resource "google_service_account_iam_member" "actor_workload_identity" {
  service_account_id = google_service_account.actor.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/${var.k8s_actor_sa}]"
}

resource "google_service_account_iam_member" "trainer_workload_identity" {
  service_account_id = google_service_account.trainer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/${var.k8s_trainer_sa}]"
}

resource "google_service_account_iam_member" "web_workload_identity" {
  service_account_id = google_service_account.web.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/${var.k8s_web_sa}]"
}

# Artifact Registry Reader for all service accounts (to pull images)
resource "google_project_iam_member" "actor_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.actor.email}"
}

resource "google_project_iam_member" "trainer_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_project_iam_member" "web_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.web.email}"
}
