variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "models_bucket_name" {
  description = "Name of the GCS bucket for models"
  type        = string
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for workloads"
  type        = string
  default     = "cartridge"
}

variable "k8s_actor_sa" {
  description = "Kubernetes service account name for Actor"
  type        = string
  default     = "cartridge-actor"
}

variable "k8s_trainer_sa" {
  description = "Kubernetes service account name for Trainer"
  type        = string
  default     = "cartridge-trainer"
}

variable "k8s_web_sa" {
  description = "Kubernetes service account name for Web"
  type        = string
  default     = "cartridge-web"
}
