source ~/.setupvars.sh
cd $BLOG_ROOT && git fetch && git rebase && git push && source venv/bin/activate && mkdocs build
python add_meta.py
# ./scripts/image_resize.sh
