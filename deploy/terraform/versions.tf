terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # Remote state. The bucket is the one manual prerequisite (see deploy/README.md):
  # Terraform cannot create the bucket that holds its own state. A backend block
  # cannot use variables, so the name is written out here (and the project id is a
  # constant in deploy/deploy.ps1 for the same reason).
  backend "gcs" {
    bucket = "project-996bb72d-b709-46ca-bbc-tfstate"
    prefix = "ai-agent-lab"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
