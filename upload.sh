# if no coscmd command, prompt the user to install it
if ! command -v coscmd &> /dev/null
then
    echo "coscmd could not be found"
    echo "Please run: pip install coscmd"
    exit
fi

coscmd upload -rs ./site /
