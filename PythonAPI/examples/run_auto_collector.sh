#!/bin/sh




SERVICE="CarlaUE4"
../../CarlaUE4.sh &
sleep 10
while true
do
	sleep 5
	if pgrep "$SERVICE" >/dev/null
	then
		echo "$SERVICE is running"
	else
		echo "$SERVICE is  stopped"
		../../CarlaUE4.sh & sleep 15	
	fi

	python ../util/config.py --reload
						
	python auto_collector.py -a --town Town05

				
done

