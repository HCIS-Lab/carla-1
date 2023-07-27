#!/bin/bash
# This shell is going to create the video for each scenario_ID

# kill al carla server 
killall -9 -r CarlaUE4-Linux

echo "Input the scenario_type you want to process"
echo "Choose from the following options:"


echo "1 - interactive"
echo "2 - non-interactive"
echo "3 - obstacle"
echo "4 - collision"

ds_id=1
read -p "Enter scenario type ID to create a data video: " ds_id
scenario_type="interactive"
if [ ${ds_id} == 1 ]
then
    scenario_type="interactive"
elif [ ${ds_id} == 2 ]
then
    scenario_type="non-interactive"
elif [ ${ds_id} == 3 ]
then
    scenario_type="obstacle"
elif [ ${ds_id} == 4 ]
then
    scenario_type="collision"
else
    echo "Invalid ID!!!"
    echo "run default setting : interactive"
fi


weather=('ClearNoon' 'CloudyNoon' 'WetNoon' 'WetCloudyNoon' 'MidRainyNoon' 'HardRainNoon' 'SoftRainNoon'
	'ClearSunset' 'CloudySunset' 'WetSunset' 'WetCloudySunset' 'MidRainSunset' 'HardRainSunset' 'SoftRainSunset'
	'ClearNight' 'CloudyNight' 'WetNight' 'WetCloudyNight' 'MidRainyNight' 'HardRainNight' 'SoftRainNight')
random_actor=('low' 'mid' 'high')

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



len=${#scenario_type}
len=$((len + 19))
folder=`ls -d ./data_collection/${scenario_type}/*`

../../CarlaUE4.sh &
sleep 10
SERVICE="CarlaUE4"
for eachfile in $folder
do

	for((i=0; i<${#weather[@]}; i++))
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
            
            echo ${eachfile:$len} 

            if [ `echo ${eachfile:$len:2} | awk -v tem="10" '{print($1==tem)? "1":"0"}'` -eq "1" ]
            then
                python data_generator.py --scenario_type ${scenario_type} --scenario_id ${eachfile:$len} --map Town10HD --no_save --generate_random_seed --weather ${weather[i]} # --random_actors ${random_actor[j]}
            else
                python data_generator.py --scenario_type ${scenario_type} --scenario_id ${eachfile:$len} --map Town0${eachfile:$len:1} --no_save --generate_random_seed --weather ${weather[i]} #  --random_actors ${random_actor[j]}
            fi

            sleep 2
        done
    done
    killall -9 -r CarlaUE4-Linux
done
