variable "prefix" {
  description = "Prefix for AWS created resources."
  type        = string
  default     = "predictions"
}

variable "port" {
  type    = number
  default = 8080
}

variable "app_runner_cpu" {
  type    = string
  default = "512"
}

variable "app_runner_memory" {
  type    = string
  default = "1024"
}

variable "ecr_repo_url" {
  type = string
}

variable "tag" {
  type    = string
  default = "latest"
}

variable "artifacts_bucket_name" {
  type = string
}

variable "model_name" {
  type    = string
  default = "sk-learn-linear-regression-reg-model"
}

variable "model_version_number" {
  type    = string
  default = "1"
}

variable "mlflow_password" {
  type = string
}

variable "mlflow_username" {
  type = string
}

variable "mlflow_url" {
  type = string
}
