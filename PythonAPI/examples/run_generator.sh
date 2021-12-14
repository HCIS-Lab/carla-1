#!/bin/sh
num_scenario=0

# features
num_weather=6
declare -a weather

# weather[0] = 'ClearNoon'
map=('Town05')
weather=('ClearNoon' 'CloudyNoon' 'WetNoon' 'WetCloudyNoon' 'MidRainyNoon' 'HardRainNoon' 'ClearSunset' 'CloudySunset' 'WetSunset' 'WetCloudySunset' 'HardRainSunset')
traj=('True' 'False')
random_actor=('True' 'False')
random_object=('True' 'False')
name=('t9-1_1_c_r_r_r' '3')
../../CarlaUE4.sh &
sleep 10
for((i=0; i<${#weather[@]}; i++))
do
        for((j=0; j<${#traj[@]}; j++))
        do
        		for((k=0; k<${#random_actor[@]}; k++))
        		do
	        			for((l=0; l<${#random_object[@]}; l++))
	        			do	
		        		python ../util/config.py --reload
		        		sleep 5
		                echo ${i}
		                time python data_generator.py -map ${map[0]} -scenario_id ${name[0]} -weather ${weather[i]} -noise_trajectory ${traj[j]} -random_actors ${random_actor[k]} -random_objects ${random_object[l]}
		                sleep 3
		            	done
        		done
        done	
done




















