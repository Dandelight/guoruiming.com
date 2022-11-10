"""
Inplementation of theme-color accroding to: https://developers.google.com/web/updates/2014/11/Support-for-theme-color-in-Chrome-39-for-Android?hl=en
"""

import os

def add_theme(file, value):
    print("processing {}".format(file))
    with open(file, "r", encoding="utf-8") as f:
        html = f.readlines()
        for idx, line in enumerate(html):
            if(line.strip() == "<head>"):
                target_idx = idx+1
        html.insert(target_idx, '<meta name="theme-color" content="{}"/>'.format(value))
    # print(html)
    with open(file, "w", encoding="utf-8") as f:
        f.writelines(html)


def add_meta():
    for root, dirs, files in os.walk("./site"):
        for file in files:
            if(file.endswith("html")):
                add_theme(os.path.join(root, file), "#ffab8f")