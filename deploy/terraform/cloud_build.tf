# `gcloud builds submit` runs the build as the default Compute Engine service account.
# In projects where it no longer gets the Editor role automatically it needs these three
# roles, otherwise the build fails right away ("does not have storage.objects.get").
#
# All bindings are additive (*_iam_member): nothing that already exists is replaced. They must exist before the image is built, so deploy.ps1
# applies them in phase 1.
locals {
  build_service_account = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"

  # Created by the first `gcloud builds submit` in the project, not owned by Terraform:
  # only its IAM policy is touched here.
  cloud_build_source_bucket = "${var.project_id}_cloudbuild"
}

# Reads the uploaded source archive.
resource "google_storage_bucket_iam_member" "build_source_reader" {
  count = var.cloud_build_source_bucket_access ? 1 : 0

  bucket = local.cloud_build_source_bucket
  role   = "roles/storage.objectViewer"
  member = local.build_service_account
}

# Pushes the image.
resource "google_artifact_registry_repository_iam_member" "build_image_writer" {
  project    = google_artifact_registry_repository.images.project
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = local.build_service_account
}

# Writes the build log to Cloud Logging.
resource "google_project_iam_member" "build_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = local.build_service_account
}
