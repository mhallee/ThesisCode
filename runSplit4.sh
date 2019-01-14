#!/bin/bash
echo "Running split4.py on all .png files in current directory"

#prompt the user for confirmation
read -p "Are you sure? " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

#ok to continue
for f in *.png; do
  echo $f
  python split4.py $f
done

