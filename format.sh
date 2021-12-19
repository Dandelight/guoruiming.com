#!/bin/sh
# FILES=$(git diff --cached --name-only --diff-filter=ACMR | sed 's| |\\ |g')


./node_modules/.bin/prettier --ignore-unknown --write ./docs
