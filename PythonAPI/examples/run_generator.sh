#!/bin/sh
num_scenario=0

# features
num_weather=6

# weather[0] = 'ClearNoon'
map=('None' 'Town01' 'Town02' 'Town03' 'Town04' 'Town05')

weather=('ClearNoon' 'CloudyNoon' 'WetNoon' 'WetCloudyNoon' 'MidRainyNoon' 'HardRainNoon' 'SoftRainNoon'
	'ClearSunset' 'CloudySunset' 'WetSunset' 'WetCloudySunset' 'MidRainSunset' 'HardRainSunset' 'SoftRainSunset'
	'ClearNight' 'CloudyNight' 'WetNight' 'WetCloudyNight' 'MidRainNight' 'HardRainNight' 'SoftRainNight')
random_actor=('low' 'mid' 'high')
name=('3_t5-5_1_p_r_r_r')

IFS='_' read -ra DES <<< "$name"

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
a=$(random_range 0 6)
b=$(random_range 7 13)
c=$(random_range 14 20)
# w=(weather[$a] weather[$b] weather[$c])
w=($a $b $c)
echo $w[*]
../../CarlaUE4.sh &
sleep 15
for((i=0; i<3; i++))
do
		for((k=0; k<${#random_actor[@]}; k++))
		do
    		python ../util/config.py --reload
    		sleep 8
            echo ${i}
            echo ${k}
            time python data_generator.py -map ${map[${DES[0]}]} -scenario_id ${name[0]} -weather ${weather[$w[$i]]} -random_actors ${random_actor[k]}
            sleep 3
		done
done




















