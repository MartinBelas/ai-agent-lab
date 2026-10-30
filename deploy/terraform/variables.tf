variable "project_id" {
  description = "GCP project (the same one that runs ai-demo)."
  type        = string
  default     = "project-996bb72d-b709-46ca-bbc"
}

variable "region" {
  description = "Region of Cloud Run and Artifact Registry."
  type        = string
  default     = "europe-west1"
}

variable "service_name" {
  description = "Name of the Cloud Run service (and of the image in Artifact Registry)."
  type        = string
  default     = "ai-agent-lab"

  # Cloud Run allows up to 49 characters, but the name is also the service account
  # id, which is limited to 6-30 characters.
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.service_name))
    error_message = "service_name must be 6-30 characters: lowercase letters, digits and hyphens, starting with a letter and not ending with a hyphen (it is used as the Cloud Run service name and as the service account id)."
  }
}

variable "image_tag" {
  description = "Tag of the image to run, normally the commit SHA. Set by deploy/deploy.ps1."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$", var.image_tag))
    error_message = "image_tag must be a valid Docker tag: at most 128 characters from A-Z a-z 0-9 _ . -, not starting with a period or a hyphen."
  }
}

variable "artifact_registry_repository" {
  description = "Name of the Docker repository in Artifact Registry."
  type        = string
  default     = "ai-agent-lab"
}

variable "images_to_keep" {
  description = "Number of most recent images kept in Artifact Registry; older ones are deleted after 30 days. Rollbacks (-SkipBuild -ImageTag) work only to images that are still kept."
  type        = number
  default     = 10

  validation {
    condition     = var.images_to_keep >= 1 && var.images_to_keep == floor(var.images_to_keep)
    error_message = "images_to_keep must be a whole number of at least 1."
  }
}

variable "ai_demo_service_account_email" {
  description = <<-EOT
    Service account of the ai-demo Cloud Run service; the only caller allowed to
    invoke this service. Empty means ai-demo@PROJECT_ID.iam.gserviceaccount.com,
    the account that ai-demo's deploy script creates; it must exist before apply.
  EOT
  type        = string
  default     = ""
}

variable "cloud_build_source_bucket_access" {
  description = "Grant the build service account read access to the Cloud Build source bucket (PROJECT_cloudbuild). deploy.ps1 sets it to false while the bucket does not exist yet; it is created by the first `gcloud builds submit`."
  type        = bool
  default     = true
}

variable "openai_secret_id" {
  description = "Existing Secret Manager secret with the OpenAI API key. It is shared with ai-demo, so Terraform only reads it and grants access; it does not own it."
  type        = string
  default     = "openai-api-key"
}

variable "tavily_secret_id" {
  description = "Secret Manager secret for the Tavily API key. Terraform creates it empty; add the value with `gcloud secrets versions add`."
  type        = string
  default     = "tavily-api-key"
}

variable "tavily_enabled" {
  description = "Mount the Tavily secret as TAVILY_API_KEY (enables the web_search tool). Set to true only after the secret has a version, otherwise the revision cannot start."
  type        = bool
  default     = false
}

variable "ai_demo_base_url" {
  description = "URL of the ai-demo service, e.g. https://ai-demo-xxxx.europe-west1.run.app (enables the get_ai_demo_providers tool). Empty disables it."
  type        = string
  default     = ""
}

variable "extra_env" {
  description = "Optional plain environment variables, e.g. OPENAI_MODEL, AGENT_MAX_STEPS, AGENT_MAX_TOOL_CALLS, AGENT_TIMEOUT_SECONDS, WEB_SEARCH_MAX_CALLS. Defaults live in app/config.py."
  type        = map(string)
  default     = {}

  validation {
    condition     = alltrue([for name in keys(var.extra_env) : !endswith(upper(name), "_API_KEY")])
    error_message = "extra_env must not contain API keys (names ending in _API_KEY): plain variables end up in the Terraform state and in the Cloud Run configuration. Secrets belong in Secret Manager (OPENAI_API_KEY and TAVILY_API_KEY are mounted from there)."
  }
}

variable "max_instances" {
  description = "Upper bound of instances; every request spends money on OpenAI, so keep it low."
  type        = number
  default     = 2

  validation {
    condition     = var.max_instances >= 1 && var.max_instances == floor(var.max_instances)
    error_message = "max_instances must be a whole number of at least 1."
  }
}

variable "request_timeout_seconds" {
  description = "Cloud Run request timeout. Must exceed AGENT_TIMEOUT_SECONDS (60 s by default) plus one slow OpenAI call."
  type        = number
  default     = 120

  validation {
    condition     = var.request_timeout_seconds >= 1 && var.request_timeout_seconds <= 3600 && var.request_timeout_seconds == floor(var.request_timeout_seconds)
    error_message = "request_timeout_seconds must be a whole number between 1 and 3600 (the Cloud Run maximum)."
  }
}
