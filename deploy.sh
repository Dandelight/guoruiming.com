source ~/.setupvars.sh
cd $BLOG_ROOT && git fetch && git rebase && git push && source venv/bin/activate && python build.py
# ./scripts/image_resize.sh
