source ~/.setupvars.sh
cd $BLOG_ROOT && git fetch && git rebase && git push && source venv/bin/activate && mkdocs build
./scripts/image_resize.sh
