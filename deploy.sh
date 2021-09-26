source ~/.setupvars.sh
cd $BLOG_PATH && git fetch && git rebase && git push && source venv/bin/activate && mkdocs build

