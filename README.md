# Mlflow tracking UI
Deploys MLFLOW tracking UI to AWS cloud. Dependencies:
- AWS App Runner to host tracking UI
- AWS RDS Postgre as mflow persistent layer
- AWS S3 as storage for storing runs artifacts

![Alt text](./img/mlflow.png)

# Deploy to AWS
1. Configure AWS credentails - https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html
2. Execute next commands:
```bash
> cd terraform/apps/mlflow
> terraform init
> terraform apply
```

This will create RDS instance, s3 bucket and App Runner serivce from Mlflow image with public endpoint (with basic-auth)

## Optional
You can create billing alarm for AWS via next steps:
```bash
> cd terraform/apps/cloudwatch
> terraform init
> terraform apply
```
This will create billing alarm for 3.33$ usage per day (approx. 100$ per month)

# Local setup
You can spin up local instance of mlflow via docker-compose.
In order to do that execute next command from the root directory of the repo:
```bash
docker-compose --file local-dev/local-docker-compose.yaml  up
```

This method does not store any persistant data, so it is only applicable for local dev/test purposes.
