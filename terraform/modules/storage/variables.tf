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

variable "enable_versioning" {
  description = "Enable GCS object versioning"
  type        = bool
  default     = true
}

variable "versions_to_keep" {
  description = "Number of object versions to keep"
  type        = number
  default     = 10
}

variable "images_to_keep" {
  description = "Number of container image versions to keep"
  type        = number
  default     = 10
}

variable "deletion_protection" {
  description = "Enable deletion protection (prevent force_destroy)"
  type        = bool
  default     = true
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default     = {}
}
