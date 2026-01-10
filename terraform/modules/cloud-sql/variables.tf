variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC network ID for private IP"
  type        = string
}

variable "private_vpc_connection" {
  description = "Private VPC connection ID (for dependency)"
  type        = string
}

variable "tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro"
}

variable "disk_size_gb" {
  description = "Initial disk size in GB"
  type        = number
  default     = 10
}

variable "high_availability" {
  description = "Enable high availability (regional)"
  type        = bool
  default     = false
}

variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "max_connections" {
  description = "Maximum database connections"
  type        = string
  default     = "100"
}

variable "enable_query_insights" {
  description = "Enable Query Insights"
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

variable "database_name" {
  description = "Name of the database to create"
  type        = string
  default     = "cartridge"
}

variable "database_user" {
  description = "Database user name"
  type        = string
  default     = "cartridge"
}

variable "database_password" {
  description = "Database user password"
  type        = string
  sensitive   = true
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default     = {}
}
