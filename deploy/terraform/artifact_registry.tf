resource "google_artifact_registry_repository" "images" {
  repository_id = var.artifact_registry_repository
  location      = var.region
  format        = "DOCKER"
  description   = "Container images of ${var.service_name}"

  # A pushed tag can never be overwritten, so a tag always means the image that was
  # built from that commit, also for a later rollback.
  docker_config {
    immutable_tags = true
  }

  # Every deploy pushes a new image; keep the newest ones and delete the rest after
  # 30 days so storage does not grow forever. KEEP takes precedence over DELETE.
  # Cleanup runs about once a day, not at the moment an image qualifies.
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-most-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.images_to_keep
    }
  }

  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "2592000s" # 30 days
    }
  }

  depends_on = [google_project_service.apis]
}
