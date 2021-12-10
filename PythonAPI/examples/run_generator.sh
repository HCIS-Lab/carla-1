#!/bin/sh
num_scenario=0

# features
num_weather=6
declare -a weather

# weather[0] = 'ClearNoon'
weather=('ClearNoon' 'CloudyNoon' 'WetNoon' 'WetCloudyNoon' 'MidRainyNoon' 'HardRainNoon' 'SoftRainNoon' 'ClearSunset' 'CloudySunset' 'WetSunset' 'WetCloudySunset' 'MidRainSunset', 'HardRainSunset' 'SoftRainSunset')
traj=('True' 'False')
random_actor=('True' 'False')
random_object=('True' 'False')
name=('aaaa' '3')
../../CarlaUE4.sh &
sleep 10
for((i=0; i<${#weather[@]}; i++))
do
        for((j=0; j<${#traj[@]}; j++))
        done
        		python ../util/config.py --reload
        		sleep 5
                echo ${i}
                time python data_generator.py -scenario_id ${name[0]} -weather ${weather[i]} -noise_trajectory ${traj[j]}
                sleep 3
        done
done




















