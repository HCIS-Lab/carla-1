echo "Input the nas location"
echo "ex: /run/user/1000/gvfs/smb-share:server=hcis_nas.local,share=carla/mini_set"
read -p "$: " path_to_nas

echo " "
echo "Input the scenario_type of data you want to retrive"
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


s=`ls -d ${path_to_nas}/${scenario_type}/*`
n="${path_to_nas}/${scenario_type}/"
len_of_path=${#n}
for basic_root in $s
do
    #echo $basic_root
    scenario_id=${basic_root:len_of_path}
    #echo $scenario_id

    output="./data_collection/${scenario_type}/${scenario_id}/" # path to store data
    root=$basic_root"/variant_scenario/"
    len=${#root}
    folder=`ls -d ${root}*`
    
    for eachfile in $folder
    do
        file_name="top"
        cp ${eachfile}/bbox/${file_name}.zip ./
        unzip ./${file_name}.zip
        rm ./${file_name}.zip
        break
    done
    cp $basic_root/scenario_description.json ${output}
    cp $basic_root/smaple.mp4 ${output}
    echo "finishing unzip file"
done

