resource "aws_s3_bucket" "this" {
  bucket = "${var.prefix}-artifacts-store"

  # Drop bucket with all content on destroy
  force_destroy = true

  tags = {
    Name = "${var.prefix}-artifacts-store"
  }
}
