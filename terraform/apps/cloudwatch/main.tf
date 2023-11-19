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
      name = "cloudwatch"
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "billing_alarm" {
  alarm_name          = "monthly-billing-alarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"
  statistic           = "Maximum"
  threshold           = "3.33"
  alarm_description   = "Alarm when AWS monthly charges exceed 100 USD"

  dimensions = {
    Currency = "USD"
  }

  treat_missing_data = "notBreaching"
}
