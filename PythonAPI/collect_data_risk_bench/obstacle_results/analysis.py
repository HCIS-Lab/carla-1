

def get_avg_distance(file_name):
    no_mask_distance = 0
    counter = 0
    f = open(f'./result_{file_name}.txt')
    for line in f.readlines():
        line = line.split("\n")[0]
        distance = line.split("#")[-1]
        # print(distance)
        counter +=1
        no_mask_distance += float(distance)
    f.close
    return no_mask_distance/counter

if __name__ == "__main__":
    file_name_list = ["mask",  "no_mask"]
    for file_name in file_name_list:
        print(file_name, get_avg_distance(file_name))


