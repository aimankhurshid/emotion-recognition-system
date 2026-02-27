#!/bin/bash

echo "================================================================================"
echo "PUSHING TO GITHUB FROM MAC (RESEARCH)"
echo "================================================================================"
echo ""

echo "Staging files..."
git add .

echo ""
echo "Git Status:"
git status -s

echo ""
read -p "Enter commit message (without tags): " commit_msg

echo ""
echo "Committing changes..."
git commit -m "$commit_msg [Mac-Research]"

echo ""
echo "Pushing to GitHub..."
git push origin main

echo ""
echo "================================================================================"
echo "PUSH COMPLETED"
echo "================================================================================"
