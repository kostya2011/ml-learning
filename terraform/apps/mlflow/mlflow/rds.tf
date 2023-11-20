resource "random_password" "this" {
  length  = 16
  special = true

  override_special = "_+=()"

  # Ignoring override_special to prevent override of existing passwords
  lifecycle {
    ignore_changes = [override_special]
  }
}


resource "aws_db_instance" "this" {
  identifier = "${var.prefix}-rds-psql"

  allocated_storage    = var.allocated_storage
  storage_type         = var.storage_type
  instance_class       = var.instance_class
  engine               = var.engine
  engine_version       = var.engine_version
  parameter_group_name = var.parameter_group_name
  publicly_accessible  = var.publicly_accessible

  vpc_security_group_ids = [aws_security_group.this.id]

  db_name  = "mlflow"
  username = "mlflow"
  password = random_password.this.result

  skip_final_snapshot     = true
  backup_retention_period = 1
  multi_az                = false
  storage_encrypted       = false
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:05:00-sun:06:00"

  tags = {
    Name = "${var.prefix}-rds-psql"
  }
}
