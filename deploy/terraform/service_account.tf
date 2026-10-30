# Identity of the running service. It gets no project-wide roles, only the
# secret accessor role on the secrets below.
resource "google_service_account" "agent" {
  account_id   = var.service_name
  display_name = "${var.service_name} Cloud Run service"

  depends_on = [google_project_service.apis]
}
