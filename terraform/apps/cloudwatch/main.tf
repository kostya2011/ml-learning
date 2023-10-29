# TODO: add aws remout backend

provider "aws" {
  region = "us-east-1"
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
