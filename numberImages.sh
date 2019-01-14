#!/bin/bash
printf "Renaming all .jpg files in current directory to %s-****.JPG\n" "$1"

#prompt the user for confirmation
read -p "Are you sure? " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

#ok to continue
a=1
for i in *.jpg; do
  new=$(printf "%s-%02d.jpg" "$1" "$a") #04 pad to length of 4
  mv -i -- "$i" "$new"
  let a=a+1
done

