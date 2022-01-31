#!/bin/sh
num_scenario=0

# features
num_weather=6

# weather[0] = 'ClearNoon'
map=('None' 'Town01' 'Town02' 'Town03' 'Town04' 'Town05')

weather=('ClearNoon' 'CloudyNoon' 'WetNoon' 'WetCloudyNoon' 'MidRainyNoon' 'HardRainNoon' 'SoftRainNoon'
	'ClearSunset' 'CloudySunset' 'WetSunset' 'WetCloudySunset' 'MidRainSunset' 'HardRainSunset' 'SoftRainSunset'
	'ClearNight' 'CloudyNight' 'WetNight' 'WetCloudyNight' 'MidRainyNight' 'HardRainNight' 'SoftRainNight')
random_actor=('low' 'mid' 'high')
#name=('3_t3-8_1_c_u_f_2_0')

#IFS='_' read -ra DES <<< "$name"

function random_range()
{
    if [ "$#" -lt "2" ]; then
        echo "Usage: random_range <low> <high>"
        return
    fi
    low=$1
    range=$(($2 - $1))
    echo $(($low+$RANDOM % $range))
}


SERVICE="CarlaUE4"
folder=`ls -d ./data_collection/*`
../../CarlaUE4.sh &
sleep 15
for scenario_name in $folder
do
	for((i=0; i<3; i++))
	do
			for((j=0; j<${#random_actor[@]}; j++))
			do
					if pgrep "$SERVICE" >/dev/null
					then
						echo "$SERVICE is running"
					else
						echo "$SERVICE is  stopped"
						../../CarlaUE4.sh & sleep 15	
					fi
					a=$(random_range 0 6)
					b=$(random_range 7 13)
					c=$(random_range 14 20)
					w=($a $b $c)

					python ../util/config.py --reload
					sleep 8
					echo ${i}
					echo ${k}
					echo ${scenario_name:18}
					time python data_generator.py -map Town0${scenario_name:18:1} -scenario_id ${scenario_name:18}  -weather ${weather[${w[${i}]}]} -random_actors ${random_actor[j]}
					sleep 3
			done
	done
done





















