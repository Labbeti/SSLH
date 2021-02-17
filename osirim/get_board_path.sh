
board_root="None"

case $1 in
	"CIFAR10" )
		board_root="/users/samova/elabbe/root_sslh/tensorboard/CIFAR10/default/";;
	"UBS8K" )
		board_root="/users/samova/elabbe/root_sslh/tensorboard/UBS8K/default/";;
	"ESC10" )
		board_root="/users/samova/elabbe/root_sslh/tensorboard/ESC10/default/";;
	"ESC50" )
		board_root="/users/samova/elabbe/root_sslh/tensorboard/ESC50/default/";;
	"GSC" )
		board_root="/users/samova/elabbe/root_sslh/tensorboard/GSC/default/";;
	"GSC12" )
		board_root="/users/samova/elabbe/root_sslh/tensorboard/GSC12/default/";;
	"AUDIOSET" )
		board_root="/users/samova/elabbe/root_sslh/tensorboard/AudioSet/default/";;
esac

echo "$board_root"
