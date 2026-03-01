#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import Ajv from "ajv";
import addFormats from "ajv-formats";
import yaml from "js-yaml";

const THIS_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(THIS_DIR, "..", "..");

const CONFIG_PATH = path.join(THIS_DIR, "ci.config.json");
const GITHUB_WORKFLOW_PATH = path.join(REPO_ROOT, ".github", "workflows", "ci.yml");
const CIRCLE_CONFIG_PATH = path.join(REPO_ROOT, ".circleci", "config.yml");
const TERMS_PATH = path.join(REPO_ROOT, "TERMS_OF_SERVICE.md");
const ENTROPY_POLICY_PATH = path.join(REPO_ROOT, "docs", "ENTROPY_CLAIMS_AND_EVIDENCE_POLICY.md");
const COMPLIANCE_CHECKLIST_PATH = path.join(REPO_ROOT, "docs", "LEGAL_COMPLIANCE_CHECKLIST.md");

const GITHUB_SCHEMA_URL = "https://json.schemastore.org/github-workflow.json";
const CIRCLE_SCHEMA_URL = "https://raw.githubusercontent.com/CircleCI-Public/circleci-yaml-language-server/main/schema.json";

function readUtf8(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function parseYaml(filePath) {
  const raw = readUtf8(filePath);
  const parsed = yaml.load(raw);
  if (!parsed || typeof parsed !== "object") {
    throw new Error(`Expected YAML object in ${filePath}`);
  }
  return parsed;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch schema: ${url} (${response.status})`);
  }
  return response.json();
}

function makeAjv() {
  const ajv = new Ajv({
    strict: false,
    allErrors: true,
    allowUnionTypes: true,
  });
  addFormats(ajv);
  return ajv;
}

function validateWithSchema(label, data, schema) {
  const ajv = makeAjv();
  const valid = ajv.validate(schema, data);
  if (valid) return;
  const issues = (ajv.errors || [])
    .slice(0, 20)
    .map((e) => `${e.instancePath || "/"} ${e.message}`)
    .join("\n");
  throw new Error(`${label} schema validation failed:\n${issues}`);
}

function assertManualOnlyGitHub(workflow) {
  const onField = workflow.on;
  if (!onField || typeof onField !== "object") {
    throw new Error("GitHub workflow must define an 'on' object.");
  }
  if (!Object.prototype.hasOwnProperty.call(onField, "workflow_dispatch")) {
    throw new Error("GitHub workflow contract requires workflow_dispatch trigger.");
  }
  if (Object.prototype.hasOwnProperty.call(onField, "push") || Object.prototype.hasOwnProperty.call(onField, "pull_request")) {
    throw new Error("GitHub workflow contract forbids push/pull_request triggers in manual-only mode.");
  }
}

function assertSemanticVariants(workflow, config) {
  const jobs = workflow.jobs || {};
  const expectedCount = config.semanticVariants.synonymMultipartLimits.length * config.semanticVariants.promptVersionExpressions.length;
  const semanticJobKeys = Object.keys(jobs).filter((k) => k.startsWith("semantic_variant_"));
  if (semanticJobKeys.length !== expectedCount) {
    throw new Error(`Expected ${expectedCount} semantic variant jobs, found ${semanticJobKeys.length}.`);
  }
}

function assertCircleWorkflowJobs(circleConfig, config) {
  const workflowJobs = circleConfig?.workflows?.ci?.jobs;
  if (!Array.isArray(workflowJobs)) {
    throw new Error("CircleCI contract requires workflows.ci.jobs to be an array.");
  }
  for (const job of config.jobs) {
    const exists = workflowJobs.some((entry) => {
      if (typeof entry === "string") return entry === job.id;
      return typeof entry === "object" && Object.prototype.hasOwnProperty.call(entry, job.id);
    });
    if (!exists) {
      throw new Error(`CircleCI workflow missing required job '${job.id}'.`);
    }
  }
}

function assertGithubJobs(workflow, config) {
  const jobs = workflow?.jobs;
  if (!jobs || typeof jobs !== "object") {
    throw new Error("GitHub workflow contract requires a jobs object.");
  }
  for (const job of config.jobs) {
    if (!Object.prototype.hasOwnProperty.call(jobs, job.id)) {
      throw new Error(`GitHub workflow missing required job '${job.id}'.`);
    }
  }
}

function assertIncludesAll(text, required, label) {
  for (const token of required) {
    if (!text.includes(token)) {
      throw new Error(`${label} is missing required content: '${token}'`);
    }
  }
}

function assertExcludesAll(text, forbidden, label) {
  for (const token of forbidden) {
    if (text.includes(token)) {
      throw new Error(`${label} contains prohibited content: '${token}'`);
    }
  }
}

function stripMarkdownSection(markdown, heading) {
  const pattern = new RegExp(`##\\s+${heading}[\\s\\S]*?(?=\\n##\\s|$)`, "i");
  return markdown.replace(pattern, "");
}

function assertLegalComplianceDocs() {
  const terms = readUtf8(TERMS_PATH);
  const policy = readUtf8(ENTROPY_POLICY_PATH);
  const checklist = readUtf8(COMPLIANCE_CHECKLIST_PATH);

  const policyLower = policy.toLowerCase();
  const termsLower = terms.toLowerCase();
  const checklistLower = checklist.toLowerCase();
  const policyClaimsLower = stripMarkdownSection(policyLower, "Prohibited Claim Language");

  assertIncludesAll(
    policyLower,
    [
      "improves unpredictability under measured conditions",
      "8.8.8.8:53",
      "8.8.4.4:53",
      "www.akamai.com:443",
      "127.0.0.1",
    ],
    "Entropy policy",
  );

  assertIncludesAll(
    policy,
    [
      "`probe_target`",
      "`probe_class`",
      "`timestamp_source`",
      "`sample_window_seconds`",
      "`sample_count`",
      "`timeout_seconds`",
      "`rtt_ms_series`",
      "`rtt_mean_ms`",
      "`rtt_variance_ms2`",
      "`drift_series_ms`",
      "`confidence_bounds_ms`",
      "`failure_handling`",
    ],
    "Entropy policy",
  );

  assertExcludesAll(
    policyClaimsLower,
    [
      "ensures additional entropy destabilization",
      "guarantees randomness",
      "guaranteed random",
    ],
    "Entropy policy",
  );

  assertIncludesAll(
    termsLower,
    [
      "improves unpredictability under measured conditions",
      "google dns",
      "akamai",
      "localhost",
      "contract gate",
    ],
    "Terms of service",
  );

  assertIncludesAll(
    checklistLower,
    [
      "isp bridge stability baseline",
      "google dns",
      "akamai",
      "localhost loopback",
      "release is blocked",
    ],
    "Compliance checklist",
  );
}

function assertLegalGateJob(config) {
  const hasLegalJob = config.jobs.some((job) => job.id === "legal_compliance_pact");
  if (!hasLegalJob) {
    throw new Error("CI config contract requires job 'legal_compliance_pact'.");
  }
}

async function main() {
  const config = JSON.parse(readUtf8(CONFIG_PATH));

  const githubWorkflow = parseYaml(GITHUB_WORKFLOW_PATH);
  const circleConfig = parseYaml(CIRCLE_CONFIG_PATH);

  const [githubSchema, circleSchema] = await Promise.all([
    fetchJson(GITHUB_SCHEMA_URL),
    fetchJson(CIRCLE_SCHEMA_URL),
  ]);

  validateWithSchema("GitHub workflow", githubWorkflow, githubSchema);
  validateWithSchema("CircleCI config", circleConfig, circleSchema);

  assertLegalGateJob(config);
  assertManualOnlyGitHub(githubWorkflow);
  assertGithubJobs(githubWorkflow, config);
  assertSemanticVariants(githubWorkflow, config);
  assertCircleWorkflowJobs(circleConfig, config);
  assertLegalComplianceDocs();

  console.log("CI pact checks passed:");
  console.log("- GitHub workflow matches schema and manual-only trigger contract");
  console.log("- GitHub/CircleCI both include required contract jobs");
  console.log("- CircleCI config matches schema and contains required jobs");
  console.log("- Semantic variant generation contract is satisfied");
  console.log("- Legal/compliance docs satisfy claim and evidence policy contract");
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
