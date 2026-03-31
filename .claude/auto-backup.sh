#!/bin/bash
# Auto-backup: commit and push changes to GitHub after file edits
# Excludes the data/ folder (already in .gitignore)

cd /Users/shining/Desktop/CP || exit 0

# Get the changed file path from stdin (PostToolUse JSON)
FILE_PATH=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null)

# Skip if the file is in the data/ directory or .claude/ directory
if [[ "$FILE_PATH" == */data/* ]] || [[ "$FILE_PATH" == */.claude/* ]]; then
    exit 0
fi

# Stage all changes (gitignore already excludes data/)
git add -A 2>/dev/null

# Check if there are staged changes
if git diff --cached --quiet 2>/dev/null; then
    exit 0
fi

# Get a short description of what changed
CHANGED_FILES=$(git diff --cached --name-only 2>/dev/null | head -5 | tr '\n' ', ' | sed 's/,$//')
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Commit and push
git commit -m "Auto-backup: ${CHANGED_FILES} (${TIMESTAMP})" 2>/dev/null
git push origin main 2>/dev/null

exit 0
