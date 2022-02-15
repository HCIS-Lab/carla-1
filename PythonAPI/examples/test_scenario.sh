#!/bin/bash
# This shell is going to create the video for each scenario_ID

echo "Input the scenario_type you want to process"
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
folder=`ls -d ./data_collection/${scenario_type}/*`

../../CarlaUE4.sh &
sleep 15
SERVICE="CarlaUE4"
for eachfile in $folder
do
    if pgrep "$SERVICE" >/dev/null
    then
        echo "$SERVICE is running"
    else
        echo "$SERVICE is  stopped"
        ../../CarlaUE4.sh & sleep 15	
    fi
    #echo ${scenario_type}
    echo ${eachfile:$len} 
    python data_generator.py --scenario_type ${scenario_type} -scenario_id ${eachfile:$len} -map Town0${eachfile:$len:1} --no_save
    sleep 3
    rm -r ./data_collection/${scenario_type}/${eachfile:$len}/ClearNoon_
done
