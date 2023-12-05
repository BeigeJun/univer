#!/bin/bash

create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

count=1

for img in `find $1 -maxdepth 2 \( -iname '*.jpg' -o -iname '*.png' \) -type f`; do
    new=image-$count.${img##*.}
    echo "Moving $img to $2/images/$new"
    create_directory "$2/images"
    cp "$img" "$2/images/$new"
    let count++
done

count=1

for mp3 in `find $1 -maxdepth 2 -iname '*.mp3' -type f`; do
    new=music-$count.${mp3##*.}
    echo "Moving $mp3 to $2/music/$new"
    create_directory "$2/music"
    cp "$mp3" "$2/music/$new"
    let count++
done

count=1

for mp4 in `find $1 -maxdepth 2 -iname '*.mp4' -type f`; do
    new=video-$count.${mp4##*.}
    echo "Moving $mp4 to $2/videos/$new"
    create_directory "$2/videos"
    cp "$mp4" "$2/videos/$new"
    let count++
done

count=1
for txt in `find $1 -maxdepth 2 -iname '*.txt' -type f`; do
    new=text-$count.${txt##*.}
    echo "Moving $txt to $2/texts/$new"
    create_directory "$2/texts"
    cp "$txt" "$2/texts/$new"
    let count++
done

echo "complete"
