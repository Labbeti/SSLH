
board_root="None"

case $1 in
	"CIFAR10" )
		board_root="/users/samova/elabbe/root/tensorboard/CIFAR10/default/";;
	"UBS8K" )
		board_root="/users/samova/elabbe/root/tensorboard/UBS8K/default/";;
	"ESC10" )
		board_root="/users/samova/elabbe/root/tensorboard/ESC10/default/";;
	"ESC50" )
		board_root="/users/samova/elabbe/root/tensorboard/ESC50/default/";;
	"GSC" )
		board_root="/users/samova/elabbe/root/tensorboard/GSC/default/";;
	"GSC12" )
		board_root="/users/samova/elabbe/root/tensorboard/GSC12/default/";;
	"AUDIOSET" )
		board_root="/users/samova/elabbe/root/tensorboard/AudioSet/default/";;
esac

echo "$board_root"
