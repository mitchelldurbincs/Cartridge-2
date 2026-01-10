# Development Environment - Cartridge2
# Provisions GCP infrastructure for development workloads

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # Uncomment for remote state (recommended for teams)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "cartridge/dev"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Generate a random password for the database
resource "random_password" "db_password" {
  length  = 24
  special = false
}

# Local values
locals {
  name_prefix = "cartridge-${var.environment}"
  labels = {
    environment = var.environment
    project     = "cartridge"
    managed-by  = "terraform"
  }
}

# Networking Module
module "networking" {
  source = "../../modules/networking"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix
}

# GKE Autopilot Cluster
module "gke" {
  source = "../../modules/gke"

  project_id             = var.project_id
  region                 = var.region
  name_prefix            = local.name_prefix
  vpc_id                 = module.networking.vpc_id
  subnet_id              = module.networking.subnet_id
  pods_range_name        = module.networking.pods_range_name
  services_range_name    = module.networking.services_range_name
  private_vpc_connection = module.networking.private_vpc_connection

  # Dev settings
  enable_private_cluster = false
  release_channel        = "REGULAR"
  deletion_protection    = false
  labels                 = local.labels
}

# Cloud SQL PostgreSQL
module "cloud_sql" {
  source = "../../modules/cloud-sql"

  project_id             = var.project_id
  region                 = var.region
  name_prefix            = local.name_prefix
  vpc_id                 = module.networking.vpc_id
  private_vpc_connection = module.networking.private_vpc_connection

  # Dev settings - small instance
  tier                  = "db-f1-micro"
  disk_size_gb          = 10
  high_availability     = false
  enable_backups        = true
  enable_query_insights = false
  deletion_protection   = false

  database_password = random_password.db_password.result
  labels            = local.labels
}

# Storage (GCS + Artifact Registry)
module "storage" {
  source = "../../modules/storage"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix

  # Dev settings
  enable_versioning   = true
  versions_to_keep    = 5
  images_to_keep      = 5
  deletion_protection = false
  labels              = local.labels
}

# IAM (Service Accounts + Workload Identity)
module "iam" {
  source = "../../modules/iam"

  project_id         = var.project_id
  name_prefix        = local.name_prefix
  models_bucket_name = module.storage.models_bucket_name

  # Kubernetes service account names
  k8s_namespace  = "cartridge"
  k8s_actor_sa   = "cartridge-actor"
  k8s_trainer_sa = "cartridge-trainer"
  k8s_web_sa     = "cartridge-web"
}
