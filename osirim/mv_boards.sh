
dataset="$1"
new_folder="$2"

board_path="`./board_path.sh $dataset`"

parent=`dirname $board_path`
old_dir="$board_path"
new_dir="$parent/$new_folder"

echo "Moving boards in $old_dir to $new_dir"
mkdir -p $new_dir
mv $old_dir* $new_dir
