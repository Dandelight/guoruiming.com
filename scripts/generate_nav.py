import os
import markdown
import frontmatter

def generate_nav(root_dir):
    nav = []

    # For each first level item
    for sub_item in os.listdir(root_dir):
        for dirpath, dirnames, filenames in os.walk(sub_item):
            folder_path = os.path.relpath(dirpath, sub_item)

            # Exclude the root directory itself
            if folder_path == ".":
                continue

            folder_nav = {
                folder_path: []
            }

            for file_name in filenames:
                if file_name.endswith(".md"):
                    file_path = os.path.join(folder_path, file_name)
                    file_title = get_file_title(os.path.join(sub_item, file_path))
                    folder_nav[folder_path].append({
                        "title": file_title,
                        "file": file_path
                    })

            if folder_nav[folder_path]:
                nav.append({sub_item: folder_nav})

    return nav


def get_file_title(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

        # Parse YAML front matter
        metadata, markdown_content = frontmatter.parse(content)

        # Check if the YAML front matter has a 'title' field
        if 'title' in metadata:
            return metadata['title']

        # Use Markdown parsing to extract the first level-1 heading as the title
        html = markdown.markdown(markdown_content)
        try:
            title = html.split("<h1>")[1].split("</h1>")[0]
        except IndexError:
            title = ".".join(file_path.split("/")[-1].split(".")[:-1])
        return title


if __name__ == "__main__":
    root_dir = "docs"
    nav_structure = generate_nav(root_dir)
    import yaml
    print(yaml.dump(nav_structure))
