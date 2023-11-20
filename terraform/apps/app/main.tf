provider "aws" {
  region = "us-east-1"
}

# Comment this out if you don't have access to scalr or wanna run it locally
data "terraform_remote_state" "mflow" {
  backend = "remote"

  config = {
    hostname     = "kostya.scalr.io"
    organization = "env-v0o4gil5gthrrt0nq"
    workspaces = {
      name = "mlflow"
    }
  }
}

# Comment this out if you don't have access to scalr or wanna run it locally
terraform {
  backend "remote" {
    hostname     = "kostya.scalr.io"
    organization = "env-v0o4gil5gthrrt0nq"

    workspaces {
      name = "predictions"
    }
  }
}

module "predictions" {
  source = "./app"

  ecr_repo_url          = data.terraform_remote_state.mflow.outputs.mlflow.ecr_url
  artifacts_bucket_name = data.terraform_remote_state.mflow.outputs.mlflow.s3_artifacts_bucket_name

  mlflow_password = data.terraform_remote_state.mflow.outputs.mlflow.mlflow_password
  mlflow_username = data.terraform_remote_state.mflow.outputs.mlflow.mlflow_user
  mlflow_url      = data.terraform_remote_state.mflow.outputs.mlflow.mflow_url
}
