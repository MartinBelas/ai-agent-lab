# Deploying to Cloud Run

The whole infrastructure of `ai-agent-lab` is described in Terraform (`deploy/terraform`) and
deployed with one command from Windows PowerShell. Nothing is clicked together in the console.

What Terraform manages in project `project-996bb72d-b709-46ca-bbc`, region `europe-west1`:

- the required APIs (Cloud Run, Artifact Registry, Cloud Build, Secret Manager, IAM, ...),
- the Artifact Registry Docker repository `ai-agent-lab`,
- the service account `ai-agent-lab` the service runs as,
- the **private** Cloud Run service `ai-agent-lab` (no `allUsers` binding) and `roles/run.invoker`
  for one caller only: the service account of `ai-demo`,
- Secret Manager: the existing `openai-api-key` secret (shared with `ai-demo`) is only read and
  made accessible to the service account, never owned or changed; the new, empty `tavily-api-key`
  secret is owned by Terraform,
- the state in a GCS bucket (backend `gcs`).

## One-off manual steps

1. Create the bucket for the Terraform state (Terraform cannot create the bucket that holds its own
   state). The name must match `backend "gcs"` in `terraform/versions.tf`:

   ```powershell
   gcloud storage buckets create gs://project-996bb72d-b709-46ca-bbc-tfstate `
     --project project-996bb72d-b709-46ca-bbc --location europe-west1 `
     --uniform-bucket-level-access --public-access-prevention
   gcloud storage buckets update gs://project-996bb72d-b709-46ca-bbc-tfstate --versioning
   ```

2. Log in twice: Terraform uses the Application Default Credentials, `gcloud builds submit` the
   normal gcloud login:

   ```powershell
   gcloud auth login
   gcloud auth application-default login
   ```

   The account needs to be able to manage the project (e.g. Owner, as for `ai-demo`).

3. Check that the secret `openai-api-key` exists (it is the one `ai-demo` uses). Terraform fails
   with a clear error if it does not.

4. Optional, for web search: after the first deploy add the Tavily key once, then set
   `tavily_enabled = true` (see below) and deploy again. The value never passes through Terraform:

   ```powershell
   $secureKey = Read-Host 'Tavily API key' -AsSecureString
   $keyFile = Join-Path $env:TEMP 'tavily.txt'
   [IO.File]::WriteAllText($keyFile, [Net.NetworkCredential]::new('', $secureKey).Password)
   gcloud secrets versions add tavily-api-key --data-file=$keyFile --project project-996bb72d-b709-46ca-bbc
   Remove-Item $keyFile
   ```

   The key is typed at a prompt, so it does not end up in the PowerShell history file, and
   `WriteAllText` writes it without a trailing newline (which would become part of the key).

   (`tavily_enabled` stays `false` until then because a Cloud Run revision cannot start with a
   secret that has no version.)

Secrets are mounted as version `latest`, which Cloud Run resolves when a revision starts. After
a key is rotated (including `openai-api-key`, shared with `ai-demo`), run `deploy.ps1` again so a
new revision picks up the new version.

## Deploy and update

```powershell
.\deploy\deploy.ps1
```

The same command creates everything on the first run and updates it afterwards. It:

1. runs `terraform init` and `terraform apply` for only what the build needs (these must exist
   before an image can be pushed): the APIs, the Artifact Registry repository and the permissions
   of the build service account (see below),
2. builds the image with Cloud Build from the repository root and pushes it to Artifact Registry,
   tagged with the git commit SHA (uncommitted changes in the files that go into the image --
   `app/`, `requirements.txt`, `Dockerfile`, `.gcloudignore` -- give a
   `<sha>-dirty-<timestamp>` tag and a warning),
3. runs `terraform apply` for the whole infrastructure with that tag, which creates/updates the
   Cloud Run service. At the end it prints the outputs, including `service_url`.

Terraform shows its plan and asks for confirmation before each apply; `-AutoApprove` skips that.
Other parameters: `-Region`, `-ImageTag` (with `-SkipBuild` to redeploy an image that is already
pushed, e.g. a rollback). The project id is a constant in the script, not a parameter, because the
state bucket in `terraform/versions.tf` belongs to that project.

### Images in Artifact Registry

Tags are **immutable**: a pushed tag can never be overwritten, neither by this script (`-ImageTag`
with an existing tag and without `-SkipBuild` fails) nor by anyone else. A cleanup policy keeps the newest
`images_to_keep` images (10) and deletes older ones after 30 days, so a rollback works only to an
image that is still there.

### Build permissions (`cloud_build.tf`)

`gcloud builds submit` runs the build as the project's default Compute Engine service account
(`<project number>-compute@developer.gserviceaccount.com`). In projects where it does not get the
Editor role automatically the build fails at once with `... does not have storage.objects.get`. Terraform therefore grants it
three narrow, additive roles (nothing existing is replaced):

- `roles/storage.objectViewer` on the bucket `<project>_cloudbuild` (the uploaded source),
- `roles/artifactregistry.writer` on the `ai-agent-lab` repository only (pushing the image),
- `roles/logging.logWriter` on the project (the build log).

The bucket is created by the very first `gcloud builds submit` in a project and is not owned by
Terraform; only its IAM policy is touched. While it does not exist yet, `deploy.ps1` skips that one
grant (with a warning) and the first build may fail on `storage.objects.get`; run the script again
and it succeeds. Should a build ever ask for more (for example writing logs to the bucket), that is
a new role to add deliberately, not something to widen quietly.

## Configuration

Optional settings go to `deploy/terraform/terraform.tfvars` (git-ignored, start from
`terraform.tfvars.example`); `deploy.ps1` picks it up automatically.

| Variable | Meaning |
| --- | --- |
| `ai_demo_base_url` | URL of `ai-demo`; enables the `get_ai_demo_providers` tool. Empty by default. |
| `tavily_enabled` | `true` after the Tavily secret got a version; enables `web_search`. |
| `ai_demo_service_account_email` | Caller allowed to invoke the service. Default: `ai-demo@<project>.iam.gserviceaccount.com`, created by `ai-demo`'s `deploy-cloud-run.ps1`, so deploy `ai-demo` first. |
| `extra_env` | Map of optional variables: `OPENAI_MODEL`, `AGENT_MAX_STEPS`, `AGENT_MAX_TOOL_CALLS`, `AGENT_TIMEOUT_SECONDS`, `WEB_SEARCH_*`, ... (defaults in `app/config.py`). |
| `images_to_keep` | Newest images kept in Artifact Registry (10); older ones are deleted after 30 days. |
| `max_instances`, `request_timeout_seconds` | Cost cap (2) and request timeout (120 s, above the 60 s agent budget). |

### `ai_demo_base_url` (optional)

Only the `get_ai_demo_providers` tool needs it: the agent asks `ai-demo` which LLM providers it
offers. Without it the tool is simply not offered and everything else works, so it can stay empty.
The URL of `ai-demo` (also shown in the Cloud Run console on the service page):

```powershell
gcloud run services describe ai-demo --region europe-west1 --format "value(status.url)"
```

Note that `ai-demo` itself is public, so this tool calls its public `/api/llm/providers` endpoint
without any token.

## Connecting `ai-demo`

Take `service_url` from the outputs (`terraform output service_url` in `deploy/terraform`) and set
it as `AGENT_LAB_BASE_URL` in `ai-demo`, with `AGENT_LAB_ENABLED=true` and
`AGENT_LAB_AUTHENTICATION=GOOGLE_ID_TOKEN`. Once `ai-demo` runs, put its URL into
`ai_demo_base_url` here and deploy again.
