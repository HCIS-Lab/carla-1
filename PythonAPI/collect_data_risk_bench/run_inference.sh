# kill al carla server 
killall -9 -r CarlaUE4-Linux
# rm ./result.txt
sleep 5
touch result.txt

SERVICE="CarlaUE4"

echo "Which scenario you want to process"
echo "Choose from the following options:"
echo ""
echo " 0 - interactive"
echo " 1 - obstacle"
echo ""
read -p "Enter scenario type: " scenario_id

echo "Input the method id you want to process"
echo "Choose from the following options:"
echo ""
echo " 0 - No mask"
echo " 1 - Ground Truth"
echo " 2 - Random"
echo " 3 - Nearest"
echo " 4 - KalmanFilter"
echo " 5 - Social-GAN"
echo " 6 - MANTRA"
echo " 7 - QCNet"
echo " 8 - DSA-RNN"
echo " 9 - DSA-RNN-Supervised"
echo " 10 - BC single-stage"
echo " 11 - BC two-stage"
echo " "

read -p "Enter ID to run Planning-aware Evaluation Benchmark: " ds_id
echo " "

if [ ${scenario_id} == 0 ]
then
    scenario_type="interactive"
elif [ ${scenario_id} == 1 ]
then
    scenario_type="obstacle"
fi

if [ ${ds_id} == 0 ]
then
    mode="No_mask"
elif [ ${ds_id} == 1 ]
then
    mode="Ground_Truth"
elif [ ${ds_id} == 2 ]
then
    mode="Random"
elif [ ${ds_id} == 3 ]
then
    mode="Nearest"
elif [ ${ds_id} == 4 ]
then
    mode="Kalman_Filter"
elif [ ${ds_id} == 5 ]
then
    mode="Social-GAN"
elif [ ${ds_id} == 6 ]
then
    mode="MANTRA"
elif [ ${ds_id} == 7 ]
then
    mode="QCNet"
elif [ ${ds_id} == 8 ]
then
    mode="DSA-RNN"
elif [ ${ds_id} == 9 ]
then
    mode="DSA-RNN-Supervised"
elif [ ${ds_id} == 10 ]
then
    mode="BC_single-stage"
elif [ ${ds_id} == 11 ]
then
    mode="BC_two-stage"
fi

while read F  ; do

    # data format
    # interactive 10_t3-1_1_p_c_l_1_0 Town10HD ClearSunset mid 14252

    # spilt the string according to  " "
    array=(${F// / })  

    COUNTER=0
    while  true ; do 
        echo inference ${array[0]} ${array[1]} ${array[2]} ${array[3]} ${array[4]} ${array[5]}
        if grep -q "${array[0]}#${array[1]}#${array[2]}#${array[3]}#${array[4]}#${array[5]}" ./result.txt
        then
            break
        else
            let COUNTER=COUNTER+1
            if [ ${COUNTER} == 20 ]
            then
                killall -9 -r CarlaUE4-Linux
                sleep 10
            fi

            # check if carla be alive
            if pgrep "$SERVICE" >/dev/null
            then
                echo "$SERVICE is running"
            else
                echo "$SERVICE is  stopped"
                ../../CarlaUE4.sh & sleep 10
            fi
            python data_generator.py --scenario_type ${array[0]} --scenario_id ${array[1]} --map ${array[2]} --weather ${array[3]} --random_actors ${array[4]} --random_seed ${array[5]} --inference --mode $mode
        fi
    done
done <./${scenario_type}_name.txt
mv ./result.txt ./${scenario_type}_results/$mode.txt
