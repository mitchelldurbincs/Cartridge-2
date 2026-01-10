# Cloud SQL Module - PostgreSQL Database
# Creates a Cloud SQL PostgreSQL instance for the replay buffer

resource "google_sql_database_instance" "postgres" {
  name             = "${var.name_prefix}-postgres"
  project          = var.project_id
  region           = var.region
  database_version = "POSTGRES_16"

  # Prevent accidental deletion
  deletion_protection = var.deletion_protection

  settings {
    tier              = var.tier
    availability_type = var.high_availability ? "REGIONAL" : "ZONAL"
    disk_size         = var.disk_size_gb
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    # Private IP only (no public IP)
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = var.vpc_id
      enable_private_path_for_google_cloud_services = true
    }

    # Backup configuration
    backup_configuration {
      enabled                        = var.enable_backups
      start_time                     = "03:00"
      point_in_time_recovery_enabled = var.enable_backups
      backup_retention_settings {
        retained_backups = 7
      }
    }

    # Maintenance window
    maintenance_window {
      day          = 7 # Sunday
      hour         = 3
      update_track = "stable"
    }

    # Database flags for performance
    database_flags {
      name  = "max_connections"
      value = var.max_connections
    }

    # Insights for query performance (optional)
    insights_config {
      query_insights_enabled  = var.enable_query_insights
      record_application_tags = var.enable_query_insights
      record_client_address   = var.enable_query_insights
    }

    user_labels = var.labels
  }

  depends_on = [var.private_vpc_connection]
}

# Database
resource "google_sql_database" "cartridge" {
  name     = var.database_name
  project  = var.project_id
  instance = google_sql_database_instance.postgres.name
}

# Database user
resource "google_sql_user" "cartridge" {
  name     = var.database_user
  project  = var.project_id
  instance = google_sql_database_instance.postgres.name
  password = var.database_password
}
