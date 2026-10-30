locals {
  # Image name without the tag; deploy.ps1 builds and pushes it with the commit SHA as the tag.
  image_repository_url = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/${var.service_name}"
}

output "image_repository_url" {
  description = "Image name without the tag."
  value       = local.image_repository_url
}

output "service_url" {
  description = "URL of the service; use it as AGENT_LAB_BASE_URL in ai-demo (also the ID token audience)."
  value       = google_cloud_run_v2_service.agent.uri
}

output "service_account_email" {
  description = "Runtime service account of the agent."
  value       = google_service_account.agent.email
}

output "ai_demo_invoker" {
  description = "The service account allowed to call this service."
  value       = local.ai_demo_service_account_email
}

output "tavily_secret_id" {
  description = "Secret to add the Tavily key to."
  value       = google_secret_manager_secret.tavily.secret_id
}
