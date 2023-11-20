resource "random_password" "mlflow_ui_password" {
  length           = 16
  special          = true
  override_special = "_%@"

  # Ignoring override_special to prevent override of existing passwords
  lifecycle {
    ignore_changes = [override_special]
  }
}

resource "aws_apprunner_service" "mlflow_server" {
  service_name = "${var.prefix}-app"

  source_configuration {
    auto_deployments_enabled = false

    image_repository {
      image_identifier      = "public.ecr.aws/t9j8s4z8/mlflow:${var.mlflow_ui_version}"
      image_repository_type = "ECR_PUBLIC"

      image_configuration {
        port = var.mlflow_ui_port
        runtime_environment_variables = {
          "MLFLOW_ARTIFACT_URI"              = "s3://${aws_s3_bucket.this.id}"
          "MLFLOW_DB_DIALECT"                = "postgresql"
          "MLFLOW_DB_USERNAME"               = aws_db_instance.this.username
          "MLFLOW_DB_PASSWORD"               = random_password.this.result
          "MLFLOW_DB_HOST"                   = aws_db_instance.this.address
          "MLFLOW_DB_PORT"                   = aws_db_instance.this.port
          "MLFLOW_DB_DATABASE"               = aws_db_instance.this.db_name
          "MLFLOW_TRACKING_USERNAME"         = local.mlflow_user
          "MLFLOW_TRACKING_PASSWORD"         = random_password.mlflow_ui_password.result
          "MLFLOW_SQLALCHEMYSTORE_POOLCLASS" = "NullPool"
        }
      }
    }
  }

  instance_configuration {
    cpu               = var.mlflow_ui_cpu
    memory            = var.mlflow_ui_memory
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
}
