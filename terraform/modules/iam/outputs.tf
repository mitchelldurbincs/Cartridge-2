output "actor_service_account_email" {
  description = "Actor service account email"
  value       = google_service_account.actor.email
}

output "trainer_service_account_email" {
  description = "Trainer service account email"
  value       = google_service_account.trainer.email
}

output "web_service_account_email" {
  description = "Web service account email"
  value       = google_service_account.web.email
}

output "actor_service_account_name" {
  description = "Actor service account name (for IAM bindings)"
  value       = google_service_account.actor.name
}

output "trainer_service_account_name" {
  description = "Trainer service account name (for IAM bindings)"
  value       = google_service_account.trainer.name
}

output "web_service_account_name" {
  description = "Web service account name (for IAM bindings)"
  value       = google_service_account.web.name
}

# Annotations for Kubernetes service accounts
output "k8s_sa_annotations" {
  description = "Annotations to add to Kubernetes service accounts for Workload Identity"
  value = {
    actor = {
      "iam.gke.io/gcp-service-account" = google_service_account.actor.email
    }
    trainer = {
      "iam.gke.io/gcp-service-account" = google_service_account.trainer.email
    }
    web = {
      "iam.gke.io/gcp-service-account" = google_service_account.web.email
    }
  }
}
