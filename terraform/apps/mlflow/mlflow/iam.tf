resource "aws_iam_role" "this" {
  name = "${var.prefix}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
        Effect = "Allow"
      },
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "tasks.apprunner.amazonaws.com"
        }
        Effect = "Allow"
      }
    ]
  })

  tags = {
    Name = "${var.prefix}-role"
  }
}

resource "aws_iam_role_policy" "this" {
  name = "${var.prefix}-policy"
  role = aws_iam_role.this.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:HeadBucket",
          "s3:ListBucketMultipartUploads",
          "s3:GetBucketTagging",
          "s3:GetObjectVersionTagging",
          "s3:ReplicateTags",
          "s3:PutObjectVersionTagging",
          "s3:ListMultipartUploadParts",
          "s3:PutObject",
          "s3:GetObject",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:AbortMultipartUpload",
          "s3:PutBucketTagging",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectTagging",
          "s3:PutObjectTagging",
          "s3:GetObjectVersion",
        ]
        Resource = [
          aws_s3_bucket.this.arn,
          "${aws_s3_bucket.this.arn}/*"
        ]
      },
    ]
  })
}
