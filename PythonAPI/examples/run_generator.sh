#!/bin/sh

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

echo "Input the scenario_type you want to geneate the data"
echo "Choose from the following options:"
echo "1 - collision"
echo "2 - obstacle"
echo "3 - interactive"
echo "4 - non-interactive"
read -p "Enter scenario type ID to create a data video: " ds_id

scenario_type="interactive"
if [ ${ds_id} == 1 ]
then
    scenario_type="collision"
elif [ ${ds_id} == 2 ]
then
    scenario_type="obstacle"
elif [ ${ds_id} == 3 ]
then
    scenario_type="interactive"
elif [ ${ds_id} == 4 ]
then
    scenario_type="non-interactive"
else
    echo "Invalid ID!!!"
    echo "run default setting : interactive"
fi

len=${#scenario_type}
len=$((len + 19))
SERVICE="CarlaUE4"
folder=`ls -d ./data_collection/${scenario_type}/*`
../../CarlaUE4.sh &
sleep 10
for scenario_name in $folder
do
	mv ./data_collection/${scenario_type}/${scenario_name:$len}/${scenario_name:$len}.mp4 ./data_collection/${scenario_type}/${scenario_name:$len}/sample.mp4 
	for((i=0; i<3; i++))
	do
			for((j=0; j<${#random_actor[@]}; j++))
			do
					sleep 5
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
					#echo ${i}
					#echo ${k}
					echo ${scenario_name:$len}

					check_name_exist=./data_collection/${scenario_type}/${scenario_name:$len}/variant_scenario/${weather[${w[${i}]}]}_${random_actor[j]}_
					#echo $check_name_exist
					if test ! -d $check_name_exist; 
					then
						if [ `echo ${scenario_name:$len:2} | awk -v tem="10" '{print($1==tem)? "1":"0"}'` -eq "1" ]
						then
							python no_rendering_mode_GT.py --scenario_type ${scenario_type} --scenario_id ${scenario_name:$len} --weather ${weather[${w[${i}]}]} --random_actors ${random_actor[j]} &
							python no_rendering_mode_Layout.py --scenario_type ${scenario_type} --scenario_id ${scenario_name:$len} --weather ${weather[${w[${i}]}]} --random_actors ${random_actor[j]} &
							
							python data_generator.py --scenario_type ${scenario_type} --scenario_id ${scenario_name:$len} --map Town10HD_opt --weather ${weather[${w[${i}]}]} --random_actors ${random_actor[j]} --save_rss
							
						else
							python no_rendering_mode_GT.py --scenario_type ${scenario_type} --scenario_id ${scenario_name:$len} --weather ${weather[${w[${i}]}]} --random_actors ${random_actor[j]} &
							python no_rendering_mode_Layout.py --scenario_type ${scenario_type} --scenario_id ${scenario_name:$len} --weather ${weather[${w[${i}]}]} --random_actors ${random_actor[j]} &
							
							python data_generator.py --scenario_type ${scenario_type} --scenario_id ${scenario_name:$len} --map Town0${scenario_name:$len:1}_opt --weather ${weather[${w[${i}]}]} --random_actors ${random_actor[j]} --save_rss

						fi
						
						mv ./data_collection/${scenario_type}/${scenario_name:$len}/${scenario_name:$len}.mp4 ./data_collection/${scenario_type}/${scenario_name:$len}/${weather[${w[${i}]}]}_${random_actor[j]}_
					else 
						echo "skip /variant_scenario/${weather[${w[${i}]}]}_${random_actor[j]}_"
					fi
			done
	done

	x="./data_collection/${scenario_type}/${scenario_name:$len}"
	f=`ls -d ${x}/*_*_`
	rm -r "${x}/timestamp"

	mkdir "${x}/variant_scenario"
	for name in $f
	do 
		FILE=${name}/finish.txt
		dynamic=${name}/dynamic_description.json
		if test -f "$FILE"; 
		then
			echo "$FILE exists."

			if test -f "$dynamic"; 
			then

				col=${name}/collision_history.json

				if [ ${ds_id} != 1 ]
				then
					if test -f "$col"; 
					then
						echo "$col exists."
						rm -r ${name}
					else
						mv ${name} "${x}/variant_scenario"
						
					fi
				else
					if test -f "$col"; 
					then
						echo "$col exists."
						mv ${name} "${x}/variant_scenario"
						
					else
						echo "$col not  exists."
						
						rm -r ${name}
						
					fi
				fi
			else
				echo "$dynamic not exist. remove this folder"
				rm -r ${name}
			fi
		else
			echo "$FILE not exist. remove this folder"
			rm -r ${name}
		fi
	done
	#./zip_data.sh ${scenario_type} ${scenario_name:$len} &	
done






















