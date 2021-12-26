#!/bin/bash
IFS=$'\n'
function read_dir() {
    for file in `ls $1`
    do
        if [ -d "$1/$file" ]  # 注意此处之间一定要加上空格，否则会报错，另外bash的注释是#
        then
            read_dir "$1/$file"
        else
            [[ "$1/$file" == *".jpg" ]] || [[ "$1/$file" == *".png" ]] && echo "Processing $1/$file" && magick "$1/$file" -resize 1000 "$1/$file" 2>>/dev/null
        fi
    done
}

read_dir site
