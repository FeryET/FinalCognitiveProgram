import os 
import json

def check_over_lapping_file_names(source_path, des_path):
    resulting_dict = {}
    for source in os.listdir(source_path):
        resulting_dict[source] = set()
        for other in os.listdir(des_path):
            others_first_part = other.split("__")[0]
            if source != other and source[:-4] == others_first_part:
                resulting_dict[source].add(other.split("__")[0])
    return resulting_dict
mainDir = "/home/farhood/Projects/datasets_of_cognitive/Data/Unprocessed Data/"
synthDir = "/home/farhood/Projects/datasets_of_cognitive/Data/AugmentedNormal/"
    
result = check_over_lapping_file_names(os.path.join(mainDir, "Cog"), os.path.join(synthDir, "Cog"))
# for k, v in result.items():
#     print("Source: {}\t Relateds: {}\n".format(k, v))
for k, v in result.items():
    if len(v) > 1:
        print(k, v)
# print(check_over_lapping_file_names(os.path.join(mainDir, "NotCog"), os.path.join(synthDir, "NotCog")))
