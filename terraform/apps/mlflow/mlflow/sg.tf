data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "this" {
  vpc_id      = data.aws_vpc.default.id
  name        = "${var.prefix}-sg"
  description = "Sg for mlflow tracking UI resources"

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
