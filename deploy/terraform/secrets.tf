# The OpenAI key is shared with ai-demo and managed outside this stack: only read
# it (to get the id) and let this service's account access it. The IAM member is
# additive, it leaves ai-demo's own access untouched.
data "google_secret_manager_secret" "openai" {
  secret_id = var.openai_secret_id

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "openai_access" {
  secret_id = data.google_secret_manager_secret.openai.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

# Owned by this stack, created empty: the value never passes through Terraform
# (and so never lands in the state). Add it once with `gcloud secrets versions add`
# (see deploy/README.md), then set tavily_enabled = true.
resource "google_secret_manager_secret" "tavily" {
  secret_id = var.tavily_secret_id

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "tavily_access" {
  secret_id = google_secret_manager_secret.tavily.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}
