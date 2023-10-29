# TODO: add aws remout backend

provider "aws" {
  region = "us-east-1"
}

module "mlflow" {
  # TODO: use remote registry? 
  source = "../../modules/mlflow"

  publicly_accessible = true
}
