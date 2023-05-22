# -*- coding=utf-8
from qcloud_cos import (CosConfig,
    CosS3Client,
    CosServiceError,
    CosClientError
)
from qcloud_cos.cos_threadpool import SimpleThreadPool

import fnmatch
import logging
from typing import (
    List,
)

import sys
import os
import gzip
from urllib.parse import urlsplit

from jinja2.exceptions import TemplateNotFound
import jinja2

from mkdocs import utils
from mkdocs.exceptions import BuildError, Abort
from mkdocs.structure.files import Files, File, _filter_paths, _sort_files
from mkdocs.structure.nav import get_navigation
import mkdocs

from mkdocs import config as mkdocs_config


# 敏感信息分离
from dotenv import load_dotenv

load_dotenv()

# 正常情况日志级别使用INFO，需要定位时可以修改为DEBUG，此时SDK会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, filename='upload.log', filemode='a')

# 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在CosConfig中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = os.getenv("QCLOUD_COS_SECRET_ID")     # 替换为用户的 SecretId，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
secret_key = os.getenv("QCLOUD_COS_SECRET_KEY")   # 替换为用户的 SecretKey，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
region = os.getenv("QCLOUD_COS_REGION")      # 替换为用户的 region，已创建桶归属的region可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
                           # COS支持的所有region列表参见https://cloud.tencent.com/document/product/436/6224
token = None               # 如果使用永久密钥不需要填入token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见https://cloud.tencent.com/document/product/436/14048
bucket = os.getenv("QCLOUD_COS_BUCKET")
uploadDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
cosBase = "/"
incremental = bool(os.getenv("BLOG_BUILD_INCREMENTAL"))

cosConfig = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token)  # 获取配置对象


def get_files(config, incremental):
    """ Walk the `docs_dir` and return a Files collection. """
    files = []
    exclude = ['.*', '/templates']
    if incremental:
        try:
            with open("last_build_finish.txt", "r") as f: timestamp = float(f.readline())
        except:
            logging.warning("NO last_build_finish.txt found")
            timestamp = 0.0
    else: timestamp = 0.0
    for source_dir, dirnames, filenames in os.walk(config['docs_dir'], followlinks=True):
        relative_dir = os.path.relpath(source_dir, config['docs_dir'])

        for dirname in list(dirnames):
            path = os.path.normpath(os.path.join(relative_dir, dirname))
            # Skip any excluded directories
            if _filter_paths(basename=dirname, path=path, is_dir=True, exclude=exclude):
                dirnames.remove(dirname)
        dirnames.sort()


        for filename in _sort_files(filenames):
            path = os.path.normpath(os.path.join(relative_dir, filename))
            # Skip any excluded files
            if _filter_paths(basename=filename, path=path, is_dir=False, exclude=exclude):
                continue
            # Skip README.md if an index file also exists in dir
            if filename.lower() == 'readme.md' and 'index.md' in filenames:
                logging.warning(f"Both index.md and readme.md found. Skipping readme.md from {source_dir}")
                continue

            if os.stat(os.path.join("docs", path)).st_mtime <= timestamp:
                logging.debug(f"Incremental build: Skip {path}")
                continue
            files.append(File(path, config['docs_dir'], config['site_dir'], config['use_directory_urls']))

    return Files(files)


def get_context(nav, files, config, page=None, base_url=''):
    """
    Return the template context for a given page or template.
    """

    if page is not None:
        base_url = utils.get_relative_url('.', page.url)

    extra_javascript = utils.create_media_urls(config['extra_javascript'], page, base_url)

    extra_css = utils.create_media_urls(config['extra_css'], page, base_url)

    if isinstance(files, Files):
        files = files.documentation_pages()

    return {
        'nav': nav,
        'pages': files,

        'base_url': base_url,

        'extra_css': extra_css,
        'extra_javascript': extra_javascript,

        'mkdocs_version': mkdocs.__version__,
        'build_date_utc': utils.get_build_datetime(),

        'config': config,
        'page': page,
    }


def _build_template(name, template, files, config, nav):
    """
    Return rendered output for given template as a string.
    """

    # Run `pre_template` plugin events.
    template = config['plugins'].run_event(
        'pre_template', template, template_name=name, config=config
    )

    if utils.is_error_template(name):
        # Force absolute URLs in the nav of error pages and account for the
        # possibility that the docs root might be different than the server root.
        # See https://github.com/mkdocs/mkdocs/issues/77.
        # However, if site_url is not set, assume the docs root and server root
        # are the same. See https://github.com/mkdocs/mkdocs/issues/1598.
        base_url = urlsplit(config['site_url'] or '/').path
    else:
        base_url = utils.get_relative_url('.', name)

    context = get_context(nav, files, config, base_url=base_url)

    # Run `template_context` plugin events.
    context = config['plugins'].run_event(
        'template_context', context, template_name=name, config=config
    )

    output = template.render(context)

    # Run `post_template` plugin events.
    output = config['plugins'].run_event(
        'post_template', output, template_name=name, config=config
    )

    return output


def _build_theme_template(template_name, env, files, config, nav):
    """ Build a template using the theme environment. """

    logging.debug(f"Building theme template: {template_name}")

    try:
        template = env.get_template(template_name)
    except TemplateNotFound:
        logging.warning(f"Template skipped: '{template_name}' not found in theme directories.")
        return

    output = _build_template(template_name, template, files, config, nav)

    if output.strip():
        output_path = os.path.join(config['site_dir'], template_name)
        utils.write_file(output.encode('utf-8'), output_path)

        if template_name == 'sitemap.xml':
            logging.debug(f"Gzipping template: {template_name}")
            gz_filename = f'{output_path}.gz'
            with open(gz_filename, 'wb') as f:
                timestamp = utils.get_build_timestamp()
                with gzip.GzipFile(fileobj=f, filename=gz_filename, mode='wb', mtime=timestamp) as gz_buf:
                    gz_buf.write(output.encode('utf-8'))
    else:
        logging.info(f"Template skipped: '{template_name}' generated empty output.")


def _build_extra_template(template_name, files, config, nav):
    """ Build user templates which are not part of the theme. """

    logging.debug(f"Building extra template: {template_name}")

    file = files.get_file_from_path(template_name)
    if file is None:
        logging.warning(f"Template skipped: '{template_name}' not found in docs_dir.")
        return

    try:
        with open(file.abs_src_path, 'r', encoding='utf-8', errors='strict') as f:
            template = jinja2.Template(f.read())
    except Exception as e:
        logging.warning(f"Error reading template '{template_name}': {e}")
        return

    output = _build_template(template_name, template, files, config, nav)

    if output.strip():
        utils.write_file(output.encode('utf-8'), file.abs_dest_path)
    else:
        logging.info(f"Template skipped: '{template_name}' generated empty output.")


def _populate_page(page, config, files, dirty=False):
    """ Read page content from docs_dir and render Markdown. """

    try:
        # When --dirty is used, only read the page if the file has been modified since the
        # previous build of the output.
        if dirty and not page.file.is_modified():
            return

        # Run the `pre_page` plugin event
        page = config['plugins'].run_event(
            'pre_page', page, config=config, files=files
        )

        page.read_source(config)

        # Run `page_markdown` plugin events.
        page.markdown = config['plugins'].run_event(
            'page_markdown', page.markdown, page=page, config=config, files=files
        )

        page.render(config, files)

        # Run `page_content` plugin events.
        page.content = config['plugins'].run_event(
            'page_content', page.content, page=page, config=config, files=files
        )
    except Exception as e:
        message = f"Error reading page '{page.file.src_path}':"
        # Prevent duplicated the error message because it will be printed immediately afterwards.
        if not isinstance(e, BuildError):
            message += f" {e}"
        logging.error(message)
        raise


def _build_page(page, config, doc_files, nav, env, dirty=False):
    """ Pass a Page to theme template and write output to site_dir. """

    try:
        # When --dirty is used, only build the page if the file has been modified since the
        # previous build of the output.
        if dirty and not page.file.is_modified():
            return

        logging.debug(f"Building page {page.file.src_path}")

        # Activate page. Signals to theme that this is the current page.
        page.active = True

        context = get_context(nav, doc_files, config, page)

        # Allow 'template:' override in md source files.
        if 'template' in page.meta:
            template = env.get_template(page.meta['template'])
        else:
            template = env.get_template('main.html')

        # Run `page_context` plugin events.
        context = config['plugins'].run_event(
            'page_context', context, page=page, config=config, nav=nav
        )

        # Render the template.
        output = template.render(context)

        # Run `post_page` plugin events.
        output = config['plugins'].run_event(
            'post_page', output, page=page, config=config
        )

        # Write the output file.
        if output.strip():
            utils.write_file(output.encode('utf-8', errors='xmlcharrefreplace'), page.file.abs_dest_path)
        else:
            logging.info(f"Page skipped: '{page.file.src_path}'. Generated empty output.")

        # Deactivate page
        page.active = False
    except Exception as e:
        message = f"Error building page '{page.file.src_path}':"
        # Prevent duplicated the error message because it will be printed immediately afterwards.
        if not isinstance(e, BuildError):
            message += f" {e}"
        logging.error(message)
        raise


def build(config, live_server=False, dirty=False, incremental=False):
    """ Perform a full site build. """
    try:
        from time import time
        start = time()

        # Run `config` plugin events.
        config = config['plugins'].run_event('config', config)

        # Run `pre_build` plugin events.
        config['plugins'].run_event('pre_build', config=config)

        if not dirty:
            logging.info("Cleaning site directory")
            utils.clean_directory(config['site_dir'])
        else:  # pragma: no cover
            # Warn user about problems that may occur with --dirty option
            logging.warning("A 'dirty' build is being performed, this will likely lead to inaccurate navigation and other"
                        " links within your site. This option is designed for site development purposes only.")

        if not live_server:  # pragma: no cover
            logging.info(f"Building documentation to directory: {config['site_dir']}")
            if dirty and site_directory_contains_stale_files(config['site_dir']):
                logging.info("The directory contains stale files. Use --clean to remove them.")

        if incremental:
            logging.warning("Using incremental build.")



        # First gather all data from all files/pages to ensure all data is consistent across all pages.

        files = get_files(config, incremental)
        env = config['theme'].get_env()
        files.add_files_from_theme(env, config)

        # Run `files` plugin events.
        files = config['plugins'].run_event('files', files, config=config)

        nav = get_navigation(files, config)

        # Run `nav` plugin events.
        nav = config['plugins'].run_event('nav', nav, config=config, files=files)

        logging.debug("Reading markdown pages.")
        for file in files.documentation_pages():
            logging.debug(f"Reading: {file.src_path}")
            _populate_page(file.page, config, files, dirty)

        # Run `env` plugin events.
        env = config['plugins'].run_event(
            'env', env, config=config, files=files
        )

        # Start writing files to site_dir now that all data is gathered. Note that order matters. Files
        # with lower precedence get written first so that files with higher precedence can overwrite them.

        logging.debug("Copying static assets.")
        files.copy_static_files(dirty=dirty)

        for template in config['theme'].static_templates:
            _build_theme_template(template, env, files, config, nav)

        for template in config['extra_templates']:
            _build_extra_template(template, files, config, nav)

        logging.debug("Building markdown pages.")
        doc_files = files.documentation_pages()
        for file in doc_files:
            _build_page(file.page, config, doc_files, nav, env, dirty)

        # Run `post_build` plugin events.
        config['plugins'].run_event('post_build', config=config)

        counts = utils.log_counter.get_counts()
        if config['strict'] and len(counts):
            msg = ', '.join([f'{v} {k.lower()}s' for k, v in counts])
            raise Abort(f'\nAborted with {msg} in strict mode!')

        logging.info('Build %d document(s) in %.2f seconds', len(doc_files), time() - start)

    except Exception as e:
        # Run `build_error` plugin events.
        config['plugins'].run_event('build_error', error=e)
        if isinstance(e, BuildError):
            logging.error(str(e))
            raise Abort('\nAborted with a BuildError!')
        raise


def site_directory_contains_stale_files(site_directory):
    """ Check if the site directory contains stale files from a previous build. """

    return True if os.path.exists(site_directory) and os.listdir(site_directory) else False


def _sort_files(filenames) -> List[str]:
    """Always sort `index` or `README` as first filename in list."""

    def key(f):
        if os.path.splitext(f)[0] in ['index', 'README']:
            return (0,)
        return (1, f)

    return sorted(filenames, key=key)


def _filter_paths(basename: str, path: str, is_dir: bool, exclude) -> bool:
    """.gitignore style file filtering."""
    for item in exclude:
        # Items ending in '/' apply only to directories.
        if item.endswith('/') and not is_dir:
            continue
        # Items starting with '/' apply to the whole path.
        # In any other cases just the basename is used.
        match = path if item.startswith('/') else basename
        if fnmatch.fnmatch(match, item.strip('/')):
            return True
    return False


def get_upload_files(dir):
    """Walk the `docs_dir` and return a Files collection."""
    files = []
    exclude = ['.*', '/templates']

    for source_dir, dirnames, filenames in os.walk(dir, followlinks=True):
        relative_dir = os.path.relpath(source_dir, dir)

        for dirname in list(dirnames):
            path = os.path.normpath(os.path.join(relative_dir, dirname))
            # Skip any excluded directories
            if _filter_paths(basename=dirname, path=path, is_dir=True, exclude=exclude):
                dirnames.remove(dirname)
        dirnames.sort()

        for filename in _sort_files(filenames):
            path = os.path.normpath(os.path.join(relative_dir, filename))
            # Skip any excluded files
            if _filter_paths(basename=filename, path=path, is_dir=False, exclude=exclude):
                continue
            files.append(path)
    return files

def do_upload(config: CosConfig, last_build_finish):
    client = CosS3Client(config)
    g = os.walk(uploadDir)
    # 创建上传的线程池
    pool = SimpleThreadPool()
    for file in get_upload_files(uploadDir):
        srcKey = os.path.join(uploadDir, file)
        if os.stat(os.path.join("site", file)).st_mtime <= timestamp:
            logging.debug(f"Incremental upload: Skip {file}")
            continue
        cosObjectKey = (cosBase + file.replace('\\', '/')).strip('/')
        pool.add_task(client.upload_file, bucket, cosObjectKey, srcKey)

    pool.wait_completion()
    result = pool.get_result()
    if not result['success_all']:
        logging.warning("Not all files upload sucessed. you should retry")
    else:
        logging.info("All files uploaded successfully.")

from scripts.add_meta import add_meta

if __name__ == '__main__':
    if incremental:
        try:
            with open("last_build_finish.txt", "r") as f: timestamp = float(f.readline())
        except:
            logging.warning("NO last_build_finish.txt found")
            incremental=False
            timestamp = 0.0
    else:
        timestamp = 0.0

    build(mkdocs_config.load_config(), dirty=incremental, incremental=incremental)

    add_meta()

    try:
        do_upload(cosConfig, timestamp)
    except CosClientError:
        logging.error("Client Error: {}".format(CosClientError))
    except CosServiceError:
        logging.error("Server Error: {}".format(CosServiceError))

    from time import time
    with open("last_build_finish.txt", "w") as f: f.writelines([str(time())])