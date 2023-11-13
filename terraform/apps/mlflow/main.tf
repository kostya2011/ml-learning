# TODO: add aws remout backend

provider "aws" {
  region = "us-east-1"
}

# Comment this out if you don't have access to scalr or wanna run it locally
terraform {
  backend "remote" {
    hostname     = "kostya.scalr.io"
    organization = "env-v0o4gil5gthrrt0nq"

    workspaces {
      name = "mlflow"
    }
  }
}

module "mlflow" {
  # TODO: use remote registry? 
  source = "./mlflow"

  publicly_accessible = true
}
