output "instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.postgres.name
}

output "instance_connection_name" {
  description = "Cloud SQL instance connection name (for Cloud SQL Proxy)"
  value       = google_sql_database_instance.postgres.connection_name
}

output "private_ip_address" {
  description = "Private IP address of the Cloud SQL instance"
  value       = google_sql_database_instance.postgres.private_ip_address
}

output "database_name" {
  description = "Database name"
  value       = google_sql_database.cartridge.name
}

output "database_user" {
  description = "Database user name"
  value       = google_sql_user.cartridge.name
}

output "connection_string" {
  description = "PostgreSQL connection string (without password)"
  value       = "postgresql://${google_sql_user.cartridge.name}@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.cartridge.name}"
  sensitive   = true
}
