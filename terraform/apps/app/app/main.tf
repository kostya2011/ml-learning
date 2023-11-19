resource "aws_apprunner_service" "this" {
  service_name = "${var.prefix}-service"

  source_configuration {
    auto_deployments_enabled = false

    authentication_configuration {
      access_role_arn = aws_iam_role.this.arn
    }

    image_repository {
      image_identifier      = "${var.ecr_repo_url}:${var.tag}"
      image_repository_type = "ECR"

      image_configuration {
        port = var.port
        runtime_environment_variables = {
          MODEL_NAME               = var.model_name
          MODEL_NUMBER             = var.model_version_number
          MLFLOW_TRACKING_PASSWORD = var.mlflow_password
          MLFLOW_TRACKING_USERNAME = var.mlflow_username
          MLFLOW_TRACKING_URI      = "https://${var.mlflow_url}"
        }
      }
    }
  }

  instance_configuration {
    cpu               = var.app_runner_cpu
    memory            = var.app_runner_memory
    instance_role_arn = aws_iam_role.this.arn
  }

  health_check_configuration {
    healthy_threshold   = 1
    unhealthy_threshold = 5
    interval            = 20
    timeout             = 20
    path                = "/health"
    protocol            = "HTTP"
  }

  tags = {
    Name = "${var.prefix}-app"
  }

  lifecycle {
    ignore_changes = [source_configuration]
  }
}
