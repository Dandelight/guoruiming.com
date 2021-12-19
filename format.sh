#!/bin/bash

DIR="./docs"

function dive() {
	local dir
	dir=$1
	echo "dir is $dir"
	for file in `ls $dir`
	do
		if [ -d $dir/$file ]
		then
			dive $dir/$file
		else
			echo $dir/$file
		fi
	done
	return
}

dive $DIR
