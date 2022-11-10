source ~/.setupvars.sh
cd $BLOG_ROOT && git fetch && git rebase && git push && source venv/bin/activate && mkdocs build &&\
        python ./scripts/add_meta.py && npx tcb hosting deploy ./site -e blog-7gq1v71gdf1f8c5a
# ./scripts/image_resize.sh

