#!/bin/sh
num_scenario=0

# features
num_weather=6

# weather[0] = 'ClearNoon'
map=('None' 'Town01' 'Town02' 'Town03' 'Town04' 'Town05')
weather=('ClearNoon' 'CloudyNoon' 'WetNoon' 'WetCloudyNoon' 'MidRainyNoon' 'HardRainNoon' 'ClearSunset' 'CloudySunset' 'WetSunset' 'WetCloudySunset' 'HardRainSunset')
traj=('False' 'True')
random_actor=('low' 'med' 'high')
random_object=('False' 'True')
name=('3_t5-5_1_p_r_r_r')

IFS='_' read -ra DES <<< "$name"

../../CarlaUE4.sh &
sleep 15
for((i=0; i<${#weather[@]}; i++))
do
        for((j=0; j<${#traj[@]}; j++))
        do
        		for((k=0; k<${#random_actor[@]}; k++))
        		do
	        			for((l=0; l<${#random_object[@]}; l++))
	        			do	
		        		python ../util/config.py --reload
		        		sleep 8
		                echo ${i}
		                echo ${j}
		                echo ${k}
		                echo ${l}
		                time python data_generator.py -map ${map[${DES[0]}]} -scenario_id ${name[0]} -weather ${weather[i]} -noise_trajectory ${traj[j]} -random_actors ${random_actor[k]} -random_objects ${random_object[l]}
		                sleep 3
		            	done
        		done
        done	
done




















