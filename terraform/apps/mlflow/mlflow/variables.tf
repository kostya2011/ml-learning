locals {
  mlflow_user = "mlflow"
}

variable "prefix" {
  description = "Prefix for AWS created resources."
  type        = string
  default     = "mlflow-tracking"
}

# DB config variables
variable "allocated_storage" {
  type    = number
  default = 5
}

variable "storage_type" {
  type    = string
  default = "gp2"
}

variable "instance_class" {
  type    = string
  default = "db.t2.micro"
}

variable "parameter_group_name" {
  type    = string
  default = "default.postgres12"
}

variable "engine" {
  type    = string
  default = "postgres"
}

variable "engine_version" {
  type    = string
  default = "12.11"
}

variable "publicly_accessible" {
  type    = bool
  default = false
}

variable "mlflow_ui_version" {
  type    = string
  default = "latest"
}

variable "mlflow_ui_port" {
  type    = number
  default = 8080
}

variable "mlflow_ui_cpu" {
  type    = string
  default = "512"
}

variable "mlflow_ui_memory" {
  type    = string
  default = "1024"
}
