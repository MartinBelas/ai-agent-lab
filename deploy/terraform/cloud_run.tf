data "google_project" "this" {
  project_id = var.project_id

  depends_on = [google_project_service.apis]
}

locals {
  image = "${local.image_repository_url}:${var.image_tag}"

  ai_demo_service_account_email = (
    var.ai_demo_service_account_email != ""
    ? var.ai_demo_service_account_email
    : "ai-demo@${var.project_id}.iam.gserviceaccount.com"
  )

  env = merge(
    var.ai_demo_base_url == "" ? {} : { AI_DEMO_BASE_URL = var.ai_demo_base_url },
    var.extra_env,
  )

  secret_env = merge(
    { OPENAI_API_KEY = data.google_secret_manager_secret.openai.secret_id },
    var.tavily_enabled ? { TAVILY_API_KEY = google_secret_manager_secret.tavily.secret_id } : {},
  )
}

resource "google_cloud_run_v2_service" "agent" {
  name     = var.service_name
  location = var.region

  # Reachable only with a Google identity token (checked by Cloud Run itself, see
  # the invoker binding below). There is deliberately no allUsers binding.
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.agent.email
    timeout         = "${var.request_timeout_seconds}s"

    # The agent is I/O bound (waits for OpenAI), a few concurrent requests per
    # instance are fine. The instance cap bounds the spend.
    max_instance_request_concurrency = 5

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    containers {
      image = local.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      startup_probe {
        http_get {
          path = "/api/health"
        }
        period_seconds    = 5
        failure_threshold = 6
      }

      dynamic "env" {
        for_each = local.env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.secret_env
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }
    }
  }

  # The revision cannot start before the service account may read the secrets.
  depends_on = [
    google_secret_manager_secret_iam_member.openai_access,
    google_secret_manager_secret_iam_member.tavily_access,
  ]

  lifecycle {
    ignore_changes = [client, client_version]
  }
}

# The only caller: the service account of ai-demo. Not authoritative, so
# invokers added by hand are not touched.
resource "google_cloud_run_v2_service_iam_member" "ai_demo_invoker" {
  project  = google_cloud_run_v2_service.agent.project
  location = google_cloud_run_v2_service.agent.location
  name     = google_cloud_run_v2_service.agent.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${local.ai_demo_service_account_email}"
}
