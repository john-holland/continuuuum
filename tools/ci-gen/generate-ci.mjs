#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import Handlebars from "handlebars";
import yaml from "js-yaml";

const THIS_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(THIS_DIR, "..", "..");

const CONFIG_PATH = path.join(THIS_DIR, "ci.config.json");
const TEMPLATE_GITHUB_PATH = path.join(THIS_DIR, "templates", "github-workflow.hbs");
const TEMPLATE_CIRCLE_PATH = path.join(THIS_DIR, "templates", "circleci-config.hbs");

function readUtf8(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function writeUtf8(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content, "utf8");
}

function normalizeNewline(text) {
  return text.endsWith("\n") ? text : `${text}\n`;
}

function isValidVersionExpression(expression) {
  const greaterThanPattern = /^[a-zA-Z_][\w-]*@>\d+(?:\.\d+)?\.x$/;
  const rangePattern = /^[a-zA-Z_][\w-]*@\{\d+(?:\.\d+)?\.x-\d+(?:\.\d+)?\.x\}$/;
  return greaterThanPattern.test(expression) || rangePattern.test(expression);
}

function validateConfig(config) {
  const requiredTopLevel = ["project", "github", "circleci", "jobs", "semanticVariants"];
  for (const key of requiredTopLevel) {
    if (!(key in config)) {
      throw new Error(`Missing required top-level key: ${key}`);
    }
  }

  if (!Array.isArray(config.jobs) || config.jobs.length === 0) {
    throw new Error("jobs must be a non-empty array");
  }

  for (const job of config.jobs) {
    if (!job.id || !job.displayName || !Array.isArray(job.steps) || job.steps.length === 0) {
      throw new Error(`Invalid job entry: ${JSON.stringify(job)}`);
    }
  }

  if (!Array.isArray(config.semanticVariants.synonymMultipartLimits) || config.semanticVariants.synonymMultipartLimits.length === 0) {
    throw new Error("semanticVariants.synonymMultipartLimits must be a non-empty array");
  }

  if (!Array.isArray(config.semanticVariants.promptVersionExpressions) || config.semanticVariants.promptVersionExpressions.length === 0) {
    throw new Error("semanticVariants.promptVersionExpressions must be a non-empty array");
  }

  for (const expression of config.semanticVariants.promptVersionExpressions) {
    if (!isValidVersionExpression(expression)) {
      throw new Error(`Invalid prompt version expression: ${expression}`);
    }
  }
}

function buildSemanticVariantMatrix(config) {
  const rows = [];
  for (const synonymLimit of config.semanticVariants.synonymMultipartLimits) {
    for (const promptExpression of config.semanticVariants.promptVersionExpressions) {
      rows.push({ synonymLimit, promptExpression });
    }
  }
  return rows;
}

function renderTemplate(templateText, context) {
  const template = Handlebars.compile(templateText, { noEscape: true });
  return normalizeNewline(template(context));
}

function validateYaml(label, content) {
  try {
    yaml.load(content);
  } catch (error) {
    throw new Error(`${label} YAML parse failed: ${error.message}`);
  }
}

function checkDrift(filePath, nextContent) {
  if (!fs.existsSync(filePath)) {
    return true;
  }
  const current = readUtf8(filePath);
  return current !== nextContent;
}

function main() {
  const args = new Set(process.argv.slice(2));
  const checkMode = args.has("--check");

  const config = JSON.parse(readUtf8(CONFIG_PATH));
  validateConfig(config);

  const context = {
    ...config,
    semanticVariantMatrix: buildSemanticVariantMatrix(config),
  };

  const githubTemplate = readUtf8(TEMPLATE_GITHUB_PATH);
  const circleTemplate = readUtf8(TEMPLATE_CIRCLE_PATH);

  const githubOutput = renderTemplate(githubTemplate, context);
  const circleOutput = renderTemplate(circleTemplate, context);

  validateYaml("GitHub workflow", githubOutput);
  validateYaml("CircleCI config", circleOutput);

  const githubPath = path.join(REPO_ROOT, config.github.workflowFile);
  const circlePath = path.join(REPO_ROOT, config.circleci.configFile);

  const githubChanged = checkDrift(githubPath, githubOutput);
  const circleChanged = checkDrift(circlePath, circleOutput);

  if (checkMode) {
    if (githubChanged || circleChanged) {
      const changedFiles = [githubChanged ? config.github.workflowFile : null, circleChanged ? config.circleci.configFile : null]
        .filter(Boolean)
        .join(", ");
      console.error(`Generated CI outputs are out of date: ${changedFiles}`);
      process.exit(1);
    }
    console.log("Generated CI outputs are up to date.");
    return;
  }

  writeUtf8(githubPath, githubOutput);
  writeUtf8(circlePath, circleOutput);

  console.log(`Generated ${config.github.workflowFile}`);
  console.log(`Generated ${config.circleci.configFile}`);
}

main();
