#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

START_DATE="2026-01-30"
END_DATE="2026-07-03"
COMMIT_COUNT=22

# Use the current branch, such as main
BRANCH="$(git branch --show-current)"

if [[ -z "$BRANCH" ]]; then
  echo "Error: You are not currently on a Git branch."
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: Run this script inside your Git repository."
  exit 1
fi

# ------------------------------------------------------------
# Convert dates to Unix timestamps
# ------------------------------------------------------------

START_SECONDS="$(date -d "${START_DATE} 00:00:00" +%s)"
END_SECONDS="$(date -d "${END_DATE} 23:59:59" +%s)"

if (( END_SECONDS <= START_SECONDS )); then
  echo "Error: END_DATE must be after START_DATE."
  exit 1
fi

TOTAL_DAYS=$(( (END_SECONDS - START_SECONDS) / 86400 + 1 ))

if (( COMMIT_COUNT > TOTAL_DAYS )); then
  echo "Error: There are not enough days for ${COMMIT_COUNT} unique commits."
  exit 1
fi

# ------------------------------------------------------------
# Commit messages
# ------------------------------------------------------------

MESSAGES=(
  "Initialize enterprise travel planner architecture"
  "Add FastAPI backend project structure"
  "Create Streamlit travel planning interface"
  "Add configuration management for local development"
  "Implement destination and itinerary data models"
  "Add Lucknow travel knowledge base structure"
  "Create document ingestion pipeline for local content"
  "Implement retrieval layer for travel information"
  "Add agentic planning workflow"
  "Connect language model service abstraction"
  "Add itinerary generation service"
  "Implement travel preference handling"
  "Add live external API integration layer"
  "Create weather and location service adapters"
  "Add hotel and attraction search support"
  "Improve RAG context assembly and grounding"
  "Add response validation and fallback handling"
  "Improve Streamlit itinerary presentation"
  "Add error handling and request logging"
  "Document local setup and environment variables"
  "Add deployment configuration and production notes"
  "Finalize enterprise travel planner documentation"
)

if (( ${#MESSAGES[@]} != COMMIT_COUNT )); then
  echo "Error: The number of messages must equal COMMIT_COUNT."
  exit 1
fi

# ------------------------------------------------------------
# Generate random unique calendar days
# This prevents all commits from appearing on one weekday.
# ------------------------------------------------------------

declare -A USED_DAYS=()
TIMESTAMPS=()

while (( ${#USED_DAYS[@]} < COMMIT_COUNT )); do
  DAY_OFFSET=$(( RANDOM % TOTAL_DAYS ))

  if [[ -n "${USED_DAYS[$DAY_OFFSET]+already_used}" ]]; then
    continue
  fi

  USED_DAYS[$DAY_OFFSET]=1

  # Random time during the selected day
  RANDOM_TIME=$(( RANDOM % 86400 ))

  COMMIT_SECONDS=$(( START_SECONDS + DAY_OFFSET * 86400 + RANDOM_TIME ))

  # Keep the timestamp inside the requested date range
  if (( COMMIT_SECONDS > END_SECONDS )); then
    COMMIT_SECONDS="$END_SECONDS"
  fi

  TIMESTAMPS+=("$COMMIT_SECONDS")
done

# Sort the timestamps chronologically
mapfile -t SORTED_TIMESTAMPS < <(
  printf '%s\n' "${TIMESTAMPS[@]}" | sort -n
)

# ------------------------------------------------------------
# Create exactly 22 commits
# Empty commits do not modify your project files.
# ------------------------------------------------------------

echo
echo "Creating ${COMMIT_COUNT} commits between ${START_DATE} and ${END_DATE}"
echo "Branch: ${BRANCH}"
echo

for (( i=0; i<COMMIT_COUNT; i++ )); do
  COMMIT_TIMESTAMP="${SORTED_TIMESTAMPS[$i]}"
  COMMIT_DATE="$(date -d "@${COMMIT_TIMESTAMP}" '+%Y-%m-%d %H:%M:%S %z')"
  MESSAGE="${MESSAGES[$i]}"

  GIT_AUTHOR_DATE="$COMMIT_DATE" \
  GIT_COMMITTER_DATE="$COMMIT_DATE" \
  git commit \
    --allow-empty \
    --quiet \
    -m "$MESSAGE"

  echo "$((i + 1)). $COMMIT_DATE - $MESSAGE"
done

echo
echo "Successfully created ${COMMIT_COUNT} commits."
echo
echo "Recent commit history:"
git log -n "$COMMIT_COUNT" \
  --date=iso \
  --pretty=format:'%h | %ad | %s'

echo
echo
echo "Current branch: ${BRANCH}"
echo
echo "To push the commits to GitHub, run:"
echo "git push -u origin ${BRANCH}"