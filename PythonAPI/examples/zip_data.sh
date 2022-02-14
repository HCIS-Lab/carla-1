!/bin/sh

scenario_type=$1
scenario_id=$2
root="./data_collection/${scenario_type}/${scenario_id}/variant_scenario/"
folder=`ls -d ${root}*`
len=${#root}

for eachfile in $folder
do
    #echo ${eachfile} 
    f=${eachfile}/depth/

    for folder_name in ${f}
    do

        # --- depth -- x6
  
        file_name="depth_back"
        zip -r ${eachfile}/depth/${file_name}.zip ${eachfile}/depth/${file_name}
        rm -r ${eachfile}/depth/${file_name}

        file_name="depth_back_left"
        zip -r ${eachfile}/depth/${file_name}.zip ${eachfile}/depth/${file_name}
        rm -r ${eachfile}/depth/${file_name}

        file_name="depth_back_right"
        zip -r ${eachfile}/depth/${file_name}.zip ${eachfile}/depth/${file_name}
        rm -r ${eachfile}/depth/${file_name}

        file_name="depth_front"
        zip -r ${eachfile}/depth/${file_name}.zip ${eachfile}/depth/${file_name}
        rm -r ${eachfile}/depth/${file_name}

        file_name="depth_left"
        zip -r ${eachfile}/depth/${file_name}.zip ${eachfile}/depth/${file_name}
        rm -r ${eachfile}/depth/${file_name}

        file_name="depth_right"
        zip -r ${eachfile}/depth/${file_name}.zip ${eachfile}/depth/${file_name}
        rm -r ${eachfile}/depth/${file_name}

        # --- dvs -- x1
        file_name="dvs"
        zip -r ${eachfile}/dvs/${file_name}.zip ${eachfile}/dvs/${file_name}
        rm -r ${eachfile}/dvs/${file_name}

        # --- optical_flow -- x1
        file_name="flow"
        zip -r ${eachfile}/optical_flow/${file_name}.zip ${eachfile}/optical_flow/${file_name}
        rm -r ${eachfile}/optical_flow/${file_name}

        # --- out_bbox   -- x1
        file_name="out_bbox"
        zip -r ${eachfile}/${file_name}.zip ${eachfile}/${file_name}
        rm -r ${eachfile}/${file_name}

        # -- out_rgb_bbox -- x1
        file_name="out_rgb_bbox"
        zip -r ${eachfile}/${file_name}.zip ${eachfile}/${file_name}
        rm -r ${eachfile}/${file_name}

        # -- ray_cast
        file_name="lidar"
        zip -r ${eachfile}/ray_cast/${file_name}.zip ${eachfile}/ray_cast/${file_name}
        rm -r ${eachfile}/ray_cast/${file_name}

        # --- rgb 
        file_name="back"
        zip -r ${eachfile}/rgb/${file_name}.zip ${eachfile}/rgb/${file_name}
        rm -r ${eachfile}/rgb/${file_name}

        file_name="back_left"
        zip -r ${eachfile}/rgb/${file_name}.zip ${eachfile}/rgb/${file_name}
        rm -r ${eachfile}/rgb/${file_name}

        file_name="back_right"
        zip -r ${eachfile}/rgb/${file_name}.zip ${eachfile}/rgb/${file_name}
        rm -r ${eachfile}/rgb/${file_name}

        file_name="front"
        zip -r ${eachfile}/rgb/${file_name}.zip ${eachfile}/rgb/${file_name}
        rm -r ${eachfile}/rgb/${file_name}

        file_name="left"
        zip -r ${eachfile}/rgb/${file_name}.zip ${eachfile}/rgb/${file_name}
        rm -r ${eachfile}/rgb/${file_name}

        file_name="right"
        zip -r ${eachfile}/rgb/${file_name}.zip ${eachfile}/rgb/${file_name}
        rm -r ${eachfile}/rgb/${file_name}

        file_name="top"
        zip -r ${eachfile}/rgb/${file_name}.zip ${eachfile}/rgb/${file_name}
        rm -r ${eachfile}/rgb/${file_name}

        # --- semantic_segmentation 
        file_name="seg_back"
        zip -r ${eachfile}/semantic_segmentation/${file_name}.zip ${eachfile}/semantic_segmentation/${file_name}
        rm -r ${eachfile}/semantic_segmentation/${file_name}

        file_name="seg_back_left"
        zip -r ${eachfile}/semantic_segmentation/${file_name}.zip ${eachfile}/semantic_segmentation/${file_name}
        rm -r ${eachfile}/semantic_segmentation/${file_name}

        file_name="seg_back_right"
        zip -r ${eachfile}/semantic_segmentation/${file_name}.zip ${eachfile}/semantic_segmentation/${file_name}
        rm -r ${eachfile}/semantic_segmentation/${file_name}

        file_name="seg_front"
        zip -r ${eachfile}/semantic_segmentation/${file_name}.zip ${eachfile}/semantic_segmentation/${file_name}
        rm -r ${eachfile}/semantic_segmentation/${file_name}

        file_name="seg_left"
        zip -r ${eachfile}/semantic_segmentation/${file_name}.zip ${eachfile}/semantic_segmentation/${file_name}
        rm -r ${eachfile}/semantic_segmentation/${file_name}

        file_name="seg_right"
        zip -r ${eachfile}/semantic_segmentation/${file_name}.zip ${eachfile}/semantic_segmentation/${file_name}
        rm -r ${eachfile}/semantic_segmentation/${file_name}

        file_name="seg_top"
        zip -r ${eachfile}/semantic_segmentation/${file_name}.zip ${eachfile}/semantic_segmentation/${file_name}
        rm -r ${eachfile}/semantic_segmentation/${file_name}

        # -- topology
        file_name="topology"
        zip -r ${eachfile}/${file_name}.zip ${eachfile}/${file_name}
        rm -r ${eachfile}/${file_name}

        # -- trajectory
        file_name="trajectory"
        zip -r ${eachfile}/${file_name}.zip ${eachfile}/${file_name}
        rm -r ${eachfile}/${file_name}

    done

    #unzip ${eachfile}/depth/depth_front.zip 

    #sleep 3
done

#./data_collection/non-interactive/3_r-1_0_0_0_f_0_0/variant_scenario/ClearNoon_mid_/depth
