# Lab: Cloud Setup for PySpark on GCP

**Course:** ETL on Cloud
**Duration:** about 90 minutes
**Level:** Beginner

In this lab you will set up a Google Cloud Platform (GCP) account, install and configure the `gcloud` CLI, create a Cloud Storage bucket, run a PySpark job on a Dataproc cluster, and clean everything up.

> **Note:** Console screens and free-trial terms change over time. If something looks different from what is written here, follow the on-screen prompts and check Google's current documentation.

---

## Learning Outcomes

By the end of this lab you will be able to:

1. Create a GCP project and link billing to it.
2. Install, initialise and authenticate the `gcloud` CLI.
3. Enable APIs and create a Cloud Storage bucket from the command line.
4. Create a Dataproc cluster and submit a PySpark job to it.
5. Delete cloud resources to avoid unwanted charges.

---

## Prerequisites

- A Google account (personal account recommended)
- A valid payment card (needed to verify identity, even for the free trial)
- A phone number for verification
- A laptop with internet access and permission to install software
- Basic comfort with a terminal (Command Prompt/PowerShell on Windows, Terminal on macOS/Linux)

## Cost Warning

Dataproc clusters and storage are billed while they exist. This lab uses a **single-node cluster** and tiny data to keep cost low. **Complete Lab 6 (Cleanup)** even if you stop early.

---

## Variables Used in This Lab

Decide these values now and replace them wherever they appear in commands.

| Placeholder | Meaning | Example |
|---|---|---|
| `PROJECT_ID` | Your GCP project ID (globally unique) | `etl-cloud-lab-12345` |
| `REGION` | Region for all resources | `asia-south1` |
| `BUCKET` | Cloud Storage bucket name (globally unique) | `etl-cloud-lab-yourname-12345` |
| `CLUSTER` | Dataproc cluster name | `lab-cluster` |

**Tip:** Use the same region for the bucket and the cluster. This avoids cross-region transfer cost and delay.

---

## Lab 1: Create the GCP Account and Project (20 min)

### Step 1.1 Sign in and start the free trial

1. Open the Google Cloud Console at `https://console.cloud.google.com` and sign in with your Google account.
2. Accept the terms and select your country.
3. Select **Start free** (or **Activate**) and enter your billing details.
4. Complete the verification steps.

### Step 1.2 Create a project

1. Click the **project picker** at the top of the Console.
2. Click **New Project**.
3. Enter a project name, for example `etl-cloud-lab`.
4. Click **Create**, then select the new project in the picker.
5. Open the project dashboard and note the **Project ID**.

### Step 1.3 Check billing

1. Open the navigation menu and go to **Billing**.
2. Confirm your project is linked to a billing account.

### Step 1.4 Set a budget alert

1. In **Billing**, open **Budgets & alerts** and click **Create budget**.
2. Select your project, set a small amount you are comfortable with, and add thresholds such as 50%, 90% and 100%.
3. Save the budget.

> Budget alerts send notifications. They do **not** stop spending automatically.

### Checkpoint 1

- [ ] I can see my project in the project picker.
- [ ] I have written down my Project ID.
- [ ] Billing is linked to the project.
- [ ] A budget alert exists.

---

## Lab 2: Install and Initialise the gcloud CLI (20 min)

> **Alternative:** If you cannot install software, click **Activate Cloud Shell** in the Console. Cloud Shell already has `gcloud` installed. You can skip Step 2.1.

### Step 2.1 Install the Google Cloud SDK

- **Windows:** Download the installer from Google's official "Install the gcloud CLI" page and run it.
- **macOS / Linux:** Follow the official instructions, or use:

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

Open a **new terminal** and verify:

```bash
gcloud --version
```

You should see the Google Cloud SDK version and a list of components.

### Step 2.2 Initialise

```bash
gcloud init
```

When prompted:

1. Choose to log in and complete sign-in in the browser.
2. Select your lab project (`PROJECT_ID`).
3. If asked, choose a default Compute Engine region and zone.

### Step 2.3 Authenticate and verify

```bash
gcloud auth login
gcloud auth list
gcloud config list
```

Expected: your account is shown as **ACTIVE**, and `project` matches your `PROJECT_ID`.

If `gcloud init` did not set the project, set it manually:

```bash
gcloud config set project PROJECT_ID
```

### Step 2.4 (Optional) Application default credentials

Used by local code and libraries that call Google APIs:

```bash
gcloud auth application-default login
```

### Step 2.5 Set default region

```bash
gcloud config set dataproc/region REGION
gcloud config set compute/region REGION
```

### Checkpoint 2

- [ ] `gcloud --version` works.
- [ ] `gcloud auth list` shows my account as active.
- [ ] `gcloud config list` shows my project and region.

---

## Lab 3: Enable APIs and Create a Bucket (10 min)

### Step 3.1 Enable required APIs

```bash
gcloud services enable dataproc.googleapis.com compute.googleapis.com storage.googleapis.com
```

This can take a minute or two. Verify:

```bash
gcloud services list --enabled | grep -E "dataproc|compute|storage"
```

(On Windows PowerShell, use `| Select-String "dataproc|compute|storage"`.)

### Step 3.2 Create a Cloud Storage bucket

```bash
gcloud storage buckets create gs://BUCKET --location=REGION
```

Bucket names must be globally unique. If you get a "name already exists" error, add random digits and retry.

### Step 3.3 Test the bucket

```bash
echo "hello cloud" > test.txt
gcloud storage cp test.txt gs://BUCKET/test/
gcloud storage ls gs://BUCKET/test/
gcloud storage cat gs://BUCKET/test/test.txt
```

### Checkpoint 3

- [ ] The three APIs are enabled.
- [ ] The bucket exists.
- [ ] I can upload and read back `test.txt`.

---

## Lab 4: Write and Upload a PySpark Job (10 min)

### Step 4.1 Create the script

Create a file named `etl_job.py` with the following content:

```python
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Output path is passed as the first argument
output_path = sys.argv[1]

spark = SparkSession.builder.appName("lab-etl-job").getOrCreate()

# Extract: small in-memory dataset (stands in for a real source)
data = [
    ("2025-01-01", "Chennai", "laptop", 2, 55000.0),
    ("2025-01-01", "Mumbai",  "phone",  5, 20000.0),
    ("2025-01-02", "Chennai", "phone",  3, 20000.0),
    ("2025-01-02", "Delhi",   "laptop", 1, 55000.0),
    ("2025-01-03", "Mumbai",  "tablet", 4, 30000.0),
    ("2025-01-03", "Chennai", "tablet", 2, 30000.0),
]
cols = ["order_date", "city", "product", "quantity", "unit_price"]
df = spark.createDataFrame(data, cols)

# Transform: compute revenue and summarise by city
df = df.withColumn("revenue", F.col("quantity") * F.col("unit_price"))
summary = (
    df.groupBy("city")
      .agg(F.sum("revenue").alias("total_revenue"),
           F.sum("quantity").alias("total_units"))
      .orderBy(F.desc("total_revenue"))
)

summary.show()

# Load: write Parquet to Cloud Storage
summary.write.mode("overwrite").parquet(output_path)

spark.stop()
```

### Step 4.2 Upload it to the bucket

```bash
gcloud storage cp etl_job.py gs://BUCKET/scripts/
gcloud storage ls gs://BUCKET/scripts/
```

### Checkpoint 4

- [ ] `etl_job.py` is visible in `gs://BUCKET/scripts/`.

---

## Lab 5: Run the Job on Dataproc (20 min)

### Step 5.1 Create a single-node cluster

```bash
gcloud dataproc clusters create CLUSTER \
  --region=REGION \
  --single-node
```

Creation usually takes a few minutes. Check the status:

```bash
gcloud dataproc clusters list --region=REGION
```

Wait until the status is **RUNNING**.

On Windows Command Prompt, put the command on one line (remove the `\` line breaks). In PowerShell, use the backtick `` ` `` for line continuation.

### Step 5.2 Submit the PySpark job

```bash
gcloud dataproc jobs submit pyspark gs://BUCKET/scripts/etl_job.py \
  --cluster=CLUSTER \
  --region=REGION \
  -- gs://BUCKET/output/city_summary
```

The arguments after `--` are passed to your script (the output path).

The terminal streams driver logs. Look for the printed summary table showing total revenue and units per city.

### Step 5.3 Check job status

```bash
gcloud dataproc jobs list --region=REGION
```

The job state should be **DONE**.

### Step 5.4 Verify the output in Cloud Storage

```bash
gcloud storage ls gs://BUCKET/output/city_summary/
```

You should see one or more `part-*.parquet` files and a `_SUCCESS` file.

### Step 5.5 Explore in the Console (optional)

1. Go to **Dataproc > Clusters** and open your cluster.
2. Open the **Jobs** tab and select your job to view its output and details.

### Checkpoint 5

- [ ] The cluster reached RUNNING.
- [ ] The job finished with state DONE.
- [ ] Parquet output exists in `gs://BUCKET/output/city_summary/`.

---

## Lab 6: Cleanup (5 min)

Do this **now** to stop charges.

### Step 6.1 Delete the cluster

```bash
gcloud dataproc clusters delete CLUSTER --region=REGION
```

Type `y` to confirm. Verify:

```bash
gcloud dataproc clusters list --region=REGION
```

### Step 6.2 Delete the bucket and its contents

```bash
gcloud storage rm --recursive gs://BUCKET
```

### Step 6.3 (Optional) Delete the project

If this project was created only for the lab:

```bash
gcloud projects delete PROJECT_ID
```

### Checkpoint 6

- [ ] No clusters are listed.
- [ ] The bucket is deleted.
- [ ] I reviewed the Billing report.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `gcloud: command not found` | Terminal not restarted or PATH not updated | Open a new terminal, or re-run the installer |
| `PERMISSION_DENIED` | Wrong account or missing IAM role | Run `gcloud auth list`, switch account, check roles in IAM |
| `API [...] not enabled` | Service not enabled for the project | Run `gcloud services enable <api>` and retry |
| Bucket name already exists | Bucket names are global | Choose a more unique name |
| Wrong project in commands | Active config points elsewhere | `gcloud config set project PROJECT_ID` |
| Cluster create fails with quota error | Regional CPU or disk quota too low | Try another region or a smaller machine type, or check Quotas in the Console |
| Cluster create fails with a network or subnet error | The project has no default network | Create a network or specify an existing subnet with `--subnet` |
| Job cannot read or write the bucket | Cluster service account lacks storage access | Check the cluster's service account roles in IAM |

---

## Final Checklist

- [ ] GCP project created with billing and a budget alert
- [ ] `gcloud` installed, initialised and authenticated
- [ ] APIs enabled and bucket created
- [ ] Dataproc cluster created and PySpark job run successfully
- [ ] Output verified in Cloud Storage
- [ ] Cluster and bucket deleted

---

## Command Quick Reference

| Task | Command |
|---|---|
| Show version | `gcloud --version` |
| Log in | `gcloud auth login` |
| List accounts | `gcloud auth list` |
| Show config | `gcloud config list` |
| Set project | `gcloud config set project PROJECT_ID` |
| Enable API | `gcloud services enable <api>` |
| Create bucket | `gcloud storage buckets create gs://BUCKET --location=REGION` |
| Copy file | `gcloud storage cp <src> <dst>` |
| Create cluster | `gcloud dataproc clusters create CLUSTER --region=REGION --single-node` |
| Submit PySpark | `gcloud dataproc jobs submit pyspark <gs://script> --cluster=CLUSTER --region=REGION` |
| Delete cluster | `gcloud dataproc clusters delete CLUSTER --region=REGION` |
| Delete bucket | `gcloud storage rm --recursive gs://BUCKET` |

---

## Reflection Questions

1. What is the difference between a region and a zone, and why did we keep the bucket and cluster in the same region?
2. Why do we delete the Dataproc cluster after the job instead of leaving it running?
3. What are the three stages of ETL in `etl_job.py`, and which lines implement each?
4. When would you use a service account instead of `gcloud auth login`?
